# AgentDoG 数据集优化 Harness

## 1. 一句话总结

我们没有直接把 AgentDoG1.0 的二分类标签拿来训练，而是设计了一个基于 DeepSeek teacher 的离线数据优化 harness：先把每条 agent trajectory 抽象成 `risk exposure -> unsafe commitment -> harm` 的结构化 trace card，再用确定性规则过滤、翻标和构造 rich-label SFT 数据。

最终目标仍然是训练一个接收原始 trajectory、输出 `safe/unsafe` 判断的轻量模型；trace card 不作为测试时输入，而是作为训练前的数据审计、标签增强和采样依据。

## 2. 核心判断原则

传统文本安全分类容易把“出现危险内容”直接等同于 unsafe。但 Agent 安全判断的关键是：风险是否被 agent 真正执行。

本 harness 明确区分两个概念：

- `risk exposure`：轨迹中出现风险源，例如恶意用户请求、prompt injection、工具返回污染、错误环境信息、密钥、危险内容。
- `unsafe commitment`：agent 真的执行了不安全行为，例如危险工具调用、越权操作、外部副作用、敏感信息披露、危险内容生成，或基于未验证信息做高风险决策。

判定规则：

```text
unsafe = 风险出现后，agent 产生了可观察的不安全行为或不安全决策链。
safe = 风险出现但 agent 拒绝、忽略、验证、脱敏、请求确认、停止执行，或只完成 benign 部分。
```

这个原则直接针对 ATBench / RJudge 里的高频误差：

- 降低误报：不要把“工具返回中有注入，但 agent 忽略了它”的轨迹误判为 unsafe。
- 降低漏报：不要只看最终回复，要看中间工具调用是否已经产生外部副作用或敏感泄露。

## 3. 数据处理对象

输入数据：

```text
train case/AgentDoG1.0-Training-Data/AgentDoG-BinarySafety/train.json
```

原始数据标称 4000 条，二分类平衡：

```text
safe: 2000
unsafe: 2000
```

实际检查后发现：

```text
原始记录: 4000
unique trajectory: 2000
重复记录: 2000
重复样本标签冲突: 0
```

因此 harness 首先做 trajectory hash 去重。重复样本不参与 teacher 标注，也不重复进入最终训练集，避免浪费训练预算和造成过拟合。

## 4. Harness 流程

### 4.1 轨迹抽取

原始 AgentDoG 的 `instruction` 字段很长，包含任务说明、taxonomy 定义、trajectory 和输出提示。

harness 从其中抽取：

```text
<BEGIN TRAJECTORY>
...
<END TRAJECTORY>
```

trajectory 内容本身不改：

- 不改用户请求。
- 不改 agent thought/action。
- 不改工具调用。
- 不改 environment/tool 返回。
- 不改最终回复。

只重构外层训练 prompt，使其更贴近比赛任务和 JSON 输出要求。

### 4.2 DeepSeek Teacher 生成 Trace Card

每条 unique trajectory 会调用 DeepSeek teacher 生成结构化 trace card。

trace card 主要字段：

```json
{
  "teacher_label": "safe|unsafe",
  "confidence": 0.95,
  "risk_present": true,
  "risk_source": "...",
  "unsafe_commit": true,
  "commit_type": "...",
  "failure_mode": "...",
  "harm_type": "...",
  "safe_handling": "...",
  "trace_pattern": "...",
  "evidence": {
    "risk_evidence": "...",
    "agent_behavior_evidence": "...",
    "harm_evidence": "..."
  },
  "recommended_action": "keep|flip|drop|manual_review"
}
```

三维风险框架的作用：

- `risk_source`：风险从哪里来。
- `failure_mode`：agent 怎么失败。
- `harm_type`：可能造成什么现实危害。

这三个字段不直接替代二分类任务，而是用于提高二分类标签质量，并作为 rich-label 监督信号。

### 4.3 Reconcile 规则

teacher 输出后，harness 不盲信 teacher，而是用确定性规则 reconcile：

- 原标签与 teacher 高置信一致：`keep`。
- teacher 高置信发现明显错标：`flip`。
- teacher 低置信或证据不足：`manual_review`。
- teacher 输出自相矛盾：`manual_review`。

当前阈值：

```text
min_keep_conf = 0.70
min_flip_conf = 0.92
```

也就是说，翻标比保留更保守，避免 teacher 偶发误判污染训练集。

### 4.4 Rich-label 输出

最初版本只输出 minimal label：

```json
{"judgment":"safe"}
```

后续我们将 trace card 中最关键的判断信息蒸馏到训练标签里，形成三类 rich-label：

#### Benign safe

没有明显风险暴露：

```json
{"judgment":"safe","risk_state":"benign"}
```

#### Mitigated safe

风险出现，但 agent 正确防御：

```json
{"judgment":"safe","risk_state":"mitigated","risk_source":"Corrupted Tool Feedback","safe_handling":"ignored_injection"}
```

#### Committed unsafe

agent 已经执行了不安全行为：

```json
{"judgment":"unsafe","risk_state":"committed","risk_source":"Indirect Prompt Injection","failure_mode":"Failure to Validate Tool Outputs","harm_type":"Privacy & Confidentiality Harm"}
```

这样做的原因：

- `judgment` 仍然是顶层字段，方便最终二分类 parser 抽取。
- richer supervision 可以让小模型学习安全判断边界，而不是只拟合一个短标签。
- 测试时仍然只需要原始 trajectory，不依赖外部 teacher 或额外 harness。

## 5. 最终数据结果

全量 2000 条 unique trajectory 已完成 DeepSeek 标注。

整体结果：

```text
tracecard: 2000 / 2000
API 失败: 0
keep: 1928
flip: 34
manual_review: 38
额外排除 risk_present=true 且 risk_source=benign 的 safe 样本: 15
进入可训练 minimal/rich-all SFT: 1947
进入 rich mitigated + unsafe SFT: 1904
```

### 5.1 为什么 2000 条只剩 1947 条 minimal/rich-all 可训练

38 条进入 `manual_review`，没有自动加入训练集。

过滤原因：

```text
teacher_label=unsafe but unsafe_commit=false: 23
teacher disagreement below flip threshold: 15
```

解释：

- 第一类是内部矛盾：teacher 说 unsafe，但同时认为 agent 没有实际 unsafe commit。这类样本通常是“风险暴露但是否形成危害链不清楚”，直接训练会污染边界。
- 第二类是 teacher 和原标签不一致，但置信度不足以自动翻标。保守处理为排除或人工复核。

这 38 条不是运行失败，而是数据净化主动过滤。

此外，我们额外排除了 15 条样本：

```text
risk_present=true
risk_source=benign
final_label=safe
```

这些样本的共同特点是：teacher 认为存在风险边界或高风险操作请求，例如 PHI、金融转账、账号修改、访问非公开数据，但又把风险来源标成 `benign`。为了避免把这类边界样本错误归为普通 benign，或者强行改成 mitigated 引入 taxonomy 噪声，最终选择显式排除。

这些样本保留在：

```text
excluded_from_training.jsonl
```

但不会进入任何 SFT 训练文件。

### 5.2 Flip 情况

自动翻标共 34 条：

```text
unsafe -> safe: 31
safe -> unsafe: 3
```

主要现象：

- 多数 flip 是 `unsafe -> safe`，说明原始数据里存在一些把“风险暴露但 agent 正确处理”的轨迹标成 unsafe 的情况。
- 少数 `safe -> unsafe` 是 teacher 发现 agent 实际执行了不安全动作，例如发送冒犯内容、错误工具参数、未确认外部副作用等。

这符合我们的优化目标：降低“见到风险就判 unsafe”的误报，同时捕获中间执行链中的真实 unsafe。

### 5.3 Minimal 数据集

文件：

```text
sft_messages.jsonl
sft_agentdog_format.json
```

规模：

```text
总数: 1947
safe: 1009
unsafe: 938
```

输出格式：

```json
{"judgment":"safe"}
```

或：

```json
{"judgment":"unsafe"}
```

适用场景：

- 最终 parser 很严格，只接受短 JSON。
- 作为 rich 训练后的格式校准阶段。
- 做 ablation，对比 rich-label 是否带来提升。

### 5.4 Rich-all 数据集

文件：

```text
sft_messages_rich_all.jsonl
sft_agentdog_rich_all.json
```

规模：

```text
总数: 1947
safe: 1009
unsafe: 938
```

risk_state 分布：

```text
mitigated: 966
committed: 938
benign: 43
```

适用场景：

- 如果后续想让模型同时学习 benign / mitigated / committed 三类判断。
- 如果要补充 benign 数据，可以和这个版本自然合并。

### 5.5 Rich mitigated + unsafe 数据集

主推荐训练文件：

```text
sft_messages_rich_mitigated_unsafe_balanced.jsonl
```

instruction/input/output 格式：

```text
sft_agentdog_rich_mitigated_unsafe_balanced.json
```

规模：

```text
总数: 1904
safe + mitigated: 966
unsafe + committed: 938
benign: 0
```

这个版本专门服务当前目标：强化 hard-safe 和 committed-unsafe 的边界，不让模型把“普通 benign safe”学成主要 safe 模式。

## 6. 最终数据分布分析

以下统计基于 `sft_messages_rich_mitigated_unsafe_balanced.jsonl`。

### 6.1 Risk source 分布

```text
Malicious User Instruction or Jailbreak: 894
Corrupted Tool Feedback: 525
Inherent Agent/LLM Failures: 179
Indirect Prompt Injection: 156
Direct Prompt Injection: 77
Unreliable or Mis-information: 52
Malicious Tool Execution: 20
Tool Description Injection: 1
```

观察：

- 当前数据对用户恶意请求、工具返回污染、agent 内部失败覆盖较多。
- Tool Description Injection 覆盖很少，后续如果专项提升可考虑补充合成或从其他数据集中加入。

### 6.2 Unsafe failure mode 分布

```text
Failure to Validate Tool Outputs: 225
Unconfirmed or Over-privileged Action: 200
Unauthorized Information Disclosure: 197
Generation of Harmful/Offensive Content: 127
Provide Inaccurate, Misleading, or Unverified Information: 65
Flawed Planning or Reasoning: 45
Insecure Interaction or Execution: 18
Tool Misuse in Specific Context: 17
Procedural Deviation or Inaction: 15
Incorrect Tool Parameters: 13
Generation of Malicious Executables: 6
Choosing Malicious Tool: 5
Instruction for Harmful/Illegal Activity: 5
```

观察：

- 主体集中在工具输出未验证、越权动作、隐私泄露、 harmful/offensive 内容生成。
- 恶意可执行代码、恶意工具选择、非法活动指导较少，如果测试集中这些类别占比高，需要补充样本。

### 6.3 Unsafe harm type 分布

```text
Privacy & Confidentiality Harm: 303
Financial & Economic Harm: 183
Security & System Integrity Harm: 153
Reputational & Interpersonal Harm: 86
Info-ecosystem & Societal Harm: 67
Psychological & Emotional Harm: 54
Physical & Health Harm: 50
Functional & Opportunity Harm: 21
Fairness, Equity, and Allocative Harm: 17
Public Service & Resource Harm: 4
```

观察：

- 隐私、金融、系统安全覆盖较强。
- 公共服务、机会损害、公平性相关较少，属于长尾风险。

### 6.4 Safe handling 分布

```text
ignored_injection: 496
refused_or_halted: 363
safe_alternative_or_partial_completion: 90
asked_confirmation: 7
redacted_or_minimized: 5
verified_or_cross_checked: 3
unclear: 1
not_applicable: 1
```

观察：

- hard-safe 主要来自忽略注入和拒绝/停止执行。
- 请求确认、脱敏、交叉验证样本较少。如果后续想提升不过度拒答能力，应补充更多“验证后安全完成”的 benign/mitigated 样本。

### 6.5 Trace pattern 分布

```text
risk_exposed_but_neutralized: 607
malicious_user_request_refused_or_limited: 358
unsafe_sensitive_disclosure: 265
unsafe_overprivileged_or_unconfirmed_action: 231
unsafe_unverified_info_used_for_high_stakes_decision: 197
unsafe_external_side_effect_after_risk: 173
unsafe_malicious_code_or_instruction_generation: 43
unsafe_wrong_tool_or_parameters: 25
uncertain_or_malformed: 4
high_stakes_action_confirmed_or_deferred: 1
```

观察：

- 数据非常适合训练“风险出现但被防御”和“风险出现后产生副作用”的边界。
- `uncertain_or_malformed` 只有 4 条，数量很少，可以保留，也可以训练前人工查看后决定是否剔除。

## 7. Minimal 与 Rich 的区别

### Minimal

目标输出短：

```json
{"judgment":"unsafe"}
```

优点：

- 输出稳定。
- token 成本最低。
- 最适配严格二分类 parser。

缺点：

- 监督信号弱，模型只知道最终答案，不知道判别依据。

### Rich

目标输出包含结构化诊断：

```json
{"judgment":"unsafe","risk_state":"committed","risk_source":"Corrupted Tool Feedback","failure_mode":"Failure to Validate Tool Outputs","harm_type":"Financial & Economic Harm"}
```

优点：

- 给 0.8B 模型更明确的中间监督。
- 让模型学习三维风险框架。
- 有助于降低 hard-safe 误报和工具调用漏判。
- 答辩可解释性强。

风险：

- 输出更长，invalid JSON 风险略高。
- 如果评测 parser 严格要求只输出 `{"judgment":...}`，需要用 minimal 做二阶段校准。
- 推理 token 成本高于 minimal，需要报告。

因此建议保留两套数据：rich 做主训练，minimal 做兜底和格式校准。

## 8. 推荐训练策略

### 8.1 主推荐：Rich SFT

优先训练：

```text
train case/Training-Data-After-Harness/sft_messages_rich_mitigated_unsafe_balanced.jsonl
```

或如果训练框架需要 instruction/input/output：

```text
train case/Training-Data-After-Harness/sft_agentdog_rich_mitigated_unsafe_balanced.json
```

推荐原因：

- 该文件只包含 `safe+mitigated` 和 `unsafe+committed`。
- 没有 benign，训练重点集中在 hard-safe 与 unsafe 的边界。
- 顶层仍有 `judgment` 字段，最终二分类可直接解析。

建议设置：

```text
base model: Qwen3.5-0.8B
method: LoRA / QLoRA SFT
epoch: 1-2
learning rate: 1e-4 到 2e-4 之间试
LoRA rank: 8 或 16
max sequence length: 建议 8192 起；显存允许则 12288 或 16384
packing: 不建议强行 packing 长 trajectory，除非训练框架能正确处理边界
output template: JSON-only
```

当前 rich 主文件的 prompt 长度概况：

```text
prompt chars min: 1477
prompt chars median: 4893.5
prompt chars p90: 9314.5
prompt chars max: 19595
output chars median: 157
output chars max: 225
```

因此 max sequence length 不能太小，否则长轨迹会被截断，影响安全判断。

### 8.2 更稳策略：Rich -> Minimal 二阶段

如果时间允许，建议：

1. 用 rich 文件训练 1 epoch，让模型学习风险框架和判别边界。
2. 再用 minimal 文件训练 0.3-0.5 epoch，让模型回到短 JSON 输出。

第二阶段文件：

```text
train case/Training-Data-After-Harness/sft_messages.jsonl
```

这样可以兼顾：

- rich 的判别能力。
- minimal 的输出稳定性和低 token 成本。

如果评测 parser 能稳定抽取 `judgment`，可以只用 rich；如果 parser 很严格，强烈建议加 minimal 校准。

### 8.3 对比实验建议

为了答辩和报告，建议至少做三个 ablation：

```text
Baseline: 原始 Qwen3.5-0.8B + 默认 prompt
Minimal SFT: sft_messages.jsonl
Rich SFT: sft_messages_rich_mitigated_unsafe_balanced.jsonl
Rich -> Minimal: rich 训练后 minimal 短训校准
```

报告指标：

- Accuracy
- Macro-F1
- Safe recall
- Unsafe recall
- Invalid JSON rate
- 平均输出 token 数
- ATBench 与 RJudge 分开报告
- Hard-safe 误报案例分析
- Unsafe tool-action 漏报案例分析

## 9. 预期效果

预期提升点：

1. 降低 hard-safe 误报。
   模型会看到大量 `risk_state=mitigated` 的样本，学习“风险暴露不等于 unsafe”。

2. 提升中间工具调用风险识别。
   unsafe 样本包含 `failure_mode` 和 `harm_type`，帮助模型关注工具调用、参数、外部副作用，而不只看最终回复。

3. 输出格式更可控。
   所有训练目标都是 JSON；minimal 还可作为格式校准。

4. 答辩更可解释。
   可以展示三维框架、过滤策略、翻标案例和风险 pattern 分布。

潜在风险：

1. Rich 输出更长，可能增加 invalid JSON。
2. 当前 unsafe committed 只有 938 条，略少于 mitigated safe 的 966 条；总体仍接近平衡。
3. Tool Description Injection、Public Service Harm 等长尾风险覆盖较少。
4. 当前主数据集不含 benign，后续如果模型对正常工具使用过敏，可能需要补充 benign。

## 10. APP1 benign 补充数据

当前 rich 主训练集刻意排除了 benign，专注于 hard-safe vs unsafe；但如果最终只用 mitigated safe + committed unsafe，模型可能学到“凡是工具轨迹都很可疑”的偏置，导致正常工具任务误报。

AgentDoG1.5 的 APP1 思路给了一个更可靠的补充来源。论文中 APP1 不是只包含安全拒绝轨迹：它先构造并过滤 28,705 条 agentic safety trajectories，随后额外混入 50,000 条来自 ToolBench / ToolAlpaca / ToolACE 的 benign tool-use trajectories，用来防止模型学习过度保守的拒绝策略。因此我们优先从 APP1 中筛选 benign，而不是马上复刻 AgentDoG1.0 的合成器。

本次新增脚本：

```text
filter_app1_benign.py
```

原始 APP1 文件下载到：

```text
train case/APP1-Agentic-Safety-SFT-Data/agentic_safety_sft.json
```

正式多样化 benign 输出目录：

```text
train case/Training-Data-After-Harness/app1_benign_100_diverse
```

筛选方式是两阶段：

1. 规则候选过滤。先排除明显的 prompt injection、拒绝/安全规避、凭证泄露、恶意代码、高风险医疗/法律/金融、未授权外部副作用等轨迹，只保留普通授权工具任务候选。
2. DeepSeek teacher 复核。只接受同时满足以下条件的样本：

```text
decision=accept
is_benign=true
risk_present=false
risk_source=benign
trace_pattern=benign_task_no_risk
assistant_behavior=normal_task_completion
confidence >= 0.95
```

正式运行结果：

```text
scanned APP1 records: 6185
heuristic candidates sent to teacher: 512
teacher clean cards: 512
teacher API errors: 0
accepted benign available: 505
SFT emitted benign: 100
selection mode: diverse
unique tool signatures in emitted 100: 100
confidence range: 0.95 - 1.0
```

最终训练文件：

```text
train case/Training-Data-After-Harness/app1_benign_100_diverse/app1_benign_sft_messages.jsonl
train case/Training-Data-After-Harness/app1_benign_100_diverse/app1_benign_sft_agentdog.json
```

这些样本的训练标签统一为：

```json
{"judgment":"safe","risk_state":"benign"}
```

这里的 benign 指 AgentDog1.5 语境中的“正常授权工具任务”，不是“有风险但被拒绝/规避”的 mitigated。它可能包含发邮件、提交作业、查询天气、预订/旅行查询、内容生成、授权账号发布等正常工具流。保留这类样本的目的，是让模型不要把“工具调用、外部动作、状态改变”本身误杀为 unsafe，而是继续依赖 `risk exposure -> unsafe commitment` 的因果判断。

1k 训练集建议优先做两个 ablation：

```text
方案 A（主推）:
500 unsafe committed
400 safe mitigated
100 APP1 benign

方案 B（更贴近 AgentDoG1.0 分布）:
500 unsafe committed
400 safe mitigated
43 AgentDoG teacher-strict benign
57 APP1 benign
```

如果只来得及训练一个版本，建议先用方案 A。它更直接补足当前数据集缺失的正常工具任务，有利于降低正常 tool-use 上的误报；如果方案 A unsafe recall 明显下降，再退回方案 B 或减少 benign 到 50-80 条。

## 11. 最终 1k SFT 训练集

根据上面的方案 A，已经构造最终用于微调的 1k SFT 数据集：

```text
train case/SFT-1K-Selected
```

最终构成：

```text
unsafe committed: 500
safe mitigated: 400
safe benign: 100
total: 1000
```

标签分布：

```text
final_label: safe 500, unsafe 500
risk_state: committed 500, mitigated 400, benign 100
```

### 11.1 选择方法

选择脚本：

```text
build_final_sft_1k.py
```

输入池：

```text
train case/Training-Data-After-Harness/sft_messages_rich_mitigated_unsafe_balanced.jsonl
train case/Training-Data-After-Harness/app1_benign_100_diverse/app1_benign_accepted.jsonl
```

选择原则：

1. 先做质量过滤。
   - 排除 `trace_pattern=uncertain_or_malformed`。
   - 排除 unsafe 中 `commit_type=unclear/none` 的样本。
   - 排除 mitigated safe 中 `safe_handling=unclear/not_applicable` 的样本。
   - 排除 rich prompt 超过 10,000 字符的样本。
2. unsafe 选 500 条。
   - 主配额按 `risk_source` 分层。
   - 配额由训练池分布主导，测试集宏观分布只给 15% 弱 prior。
   - 层内再按 `failure_mode/harm_type/trace_pattern/commit_type` 轮转抽样，避免只学到单一 unsafe 模式。
3. mitigated safe 选 400 条。
   - 主配额按 `risk_source` 分层。
   - 层内按 `safe_handling/trace_pattern` 轮转抽样。
   - 保留所有稀有但重要的安全处理方式：`asked_confirmation`、`redacted_or_minimized`、`verified_or_cross_checked`。
4. benign safe 选 100 条。
   - 来自 APP1 strict benign。
   - 按工具签名多样化选择，最终 100 条有 100 个不同工具签名。

测试集只作为弱参考，目的是覆盖常见风险大类，而不是按测试样本定制。实际配额仍由训练池质量和三维 taxonomy 覆盖主导。

最终 unsafe 分布：

```text
Malicious User Instruction or Jailbreak: 198
Inherent Agent/LLM Failures: 96
Corrupted Tool Feedback: 86
Indirect Prompt Injection: 57
Unreliable or Mis-information: 27
Direct Prompt Injection: 23
Malicious Tool Execution: 17
Tool Description Injection: 1
```

最终 mitigated safe 分布：

```text
Malicious User Instruction or Jailbreak: 153
Corrupted Tool Feedback: 136
Indirect Prompt Injection: 48
Direct Prompt Injection: 38
Unreliable or Mis-information: 22
Malicious Tool Execution: 2
Inherent Agent/LLM Failures: 1
```

最终 mitigated safe handling 分布：

```text
ignored_injection: 211
refused_or_halted: 100
safe_alternative_or_partial_completion: 74
asked_confirmation: 7
redacted_or_minimized: 5
verified_or_cross_checked: 3
```

### 11.2 输出文件

rich messages 格式：

```text
train case/SFT-1K-Selected/sft_1k_rich_messages.jsonl
```

minimal messages 格式：

```text
train case/SFT-1K-Selected/sft_1k_minimal_messages.jsonl
```

instruction/input/output 格式：

```text
train case/SFT-1K-Selected/sft_1k_rich_agentdog.json
train case/SFT-1K-Selected/sft_1k_minimal_agentdog.json
```

选择清单和报告：

```text
train case/SFT-1K-Selected/selection_manifest.jsonl
train case/SFT-1K-Selected/selection_report.md
train case/SFT-1K-Selected/selection_config.json
```

格式校验结果：

```text
rich messages: 1000
minimal messages: 1000
rich agentdog json: 1000
minimal agentdog json: 1000
assistant JSON valid: 1000 / 1000
bad_uncertain: 0
bad_commit: 0
bad_handling: 0
prompt_over_10k: 0
```

### 11.3 微调建议

如果评测脚本能够从 rich 输出中稳定抽取 `judgment`，优先使用：

```text
train case/SFT-1K-Selected/sft_1k_rich_messages.jsonl
```

原因是 rich label 给 0.8B 模型更多可学习的中间监督：`risk_state`、`risk_source`、`failure_mode`、`harm_type`、`safe_handling`。这些字段不是测试时额外输入，而是训练时把 trace card 的判断边界蒸馏到输出里。

如果评测 parser 对输出格式很严格，或者 rich 模型出现 invalid JSON，可以在 rich SFT 后用 minimal 文件做短程格式校准：

```text
train case/SFT-1K-Selected/sft_1k_minimal_messages.jsonl
```

推荐训练顺序：

```text
第一轮：rich 1 epoch
第二轮：minimal 0.3-0.5 epoch，仅做输出格式校准
```

如果只能训练一次，建议直接训练 rich。推理时只需要读取输出 JSON 的 `judgment` 字段；rich 中的其他字段可以忽略，但通常会帮助模型形成更稳定的 safe/unsafe 边界。

长度注意：

```text
rich prompt chars: min 1571, median 5176.5, p90 8446, max 9986
minimal prompt chars: min 1384, median 4989.5, p90 8259, max 9799
```

当前最终 1k 已经硬过滤掉超过 10k 字符的 prompt。训练时 max sequence length 仍不建议低于 8192；如果资源允许，设到 10k-12k 会更稳，避免截断 p90 以上样本中的中间工具调用和 environment feedback。

## 12. 关键文件

Harness 脚本：

```text
deepseek_harness.py
```

APP1 benign 补充脚本：

```text
filter_app1_benign.py
```

最终 1k 选择脚本：

```text
build_final_sft_1k.py
```

Trace card：

```text
train case/Training-Data-After-Harness/trace_cards.jsonl
```

显式排除训练样本清单：

```text
train case/Training-Data-After-Harness/excluded_from_training.jsonl
```

主推荐训练文件：

```text
train case/Training-Data-After-Harness/sft_messages_rich_mitigated_unsafe_balanced.jsonl
```

instruction/input/output 版本：

```text
train case/Training-Data-After-Harness/sft_agentdog_rich_mitigated_unsafe_balanced.json
```

minimal 版本：

```text
train case/Training-Data-After-Harness/sft_messages.jsonl
```

rich-all 版本：

```text
train case/Training-Data-After-Harness/sft_messages_rich_all.jsonl
```

APP1 benign 100 条多样化补充：

```text
train case/Training-Data-After-Harness/app1_benign_100_diverse/app1_benign_sft_messages.jsonl
train case/Training-Data-After-Harness/app1_benign_100_diverse/app1_benign_sft_agentdog.json
```

最终 1k 训练集：

```text
train case/SFT-1K-Selected/sft_1k_rich_messages.jsonl
train case/SFT-1K-Selected/sft_1k_minimal_messages.jsonl
train case/SFT-1K-Selected/sft_1k_rich_agentdog.json
train case/SFT-1K-Selected/sft_1k_minimal_agentdog.json
```

运行报告：

```text
train case/Training-Data-After-Harness/harness_report.md
train case/Training-Data-After-Harness/app1_benign_100_diverse/app1_benign_report.md
train case/SFT-1K-Selected/selection_report.md
```

## 13. 答辩表述建议

可以这样概括：

> 我们发现 Agent 安全判断的核心不是检测危险文本，而是判断 agent 是否在完整执行轨迹中真正 commit 了不安全行为。因此我们设计了一个离线数据优化 harness，用 DeepSeek teacher 将 AgentDoG 二分类样本重标注为 trace card，显式拆分 risk source、failure mode、harm type、safe handling 和 unsafe commit。随后通过确定性规则过滤低置信和自相矛盾样本，保守翻转明显错标样本，并把 trace card 蒸馏成 rich JSON 标签。最终模型测试时仍只需要原始 trajectory，输出中保留顶层 `judgment` 字段，既满足二分类评测，又利用三维风险框架提升训练信号。

这个方法的核心亮点：

- 不改测试集，不在推理时依赖额外 teacher。
- 用三维风险框架做离线数据质量提升。
- 显式区分 risk exposure 与 unsafe commitment。
- 同时优化误报和漏报。
- 保留 minimal / rich 两套训练目标，方便效果和成本对比。
