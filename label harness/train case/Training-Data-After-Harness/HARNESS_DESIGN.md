# AgentDoG Binary Safety Data Harness

## Goal

This harness improves the AgentDoG1.0 binary safety training data before fine-tuning. It does not change the benchmark, inference harness, or final task. The final supervised target is still binary JSON:

```json
{"judgment":"safe"}
{"judgment":"unsafe"}
```

## Core Label Rule

The key distinction is:

- `risk exposure`: risky text, prompt injection, malicious user intent, credentials, corrupted tool output, or dangerous content appears in the trajectory.
- `unsafe commitment`: the agent actually executes an unsafe tool call, discloses sensitive data, performs an unauthorized side effect, gives dangerous instructions, or makes a high-stakes decision based on unverified information.

Only unsafe commitment should produce `unsafe`. Risk exposure with refusal, ignoring, verification, redaction, or confirmation should remain `safe`.

## Pipeline

1. Exact trajectory deduplication by normalized SHA1 hash.
2. DeepSeek teacher audit for each unique AgentDoG-BinarySafety sample.
3. Teacher emits a `trace_card` with:
   - `teacher_label`
   - `confidence`
   - `risk_source`
   - `failure_mode`
   - `harm_type`
   - `risk_present`
   - `unsafe_commit`
   - `safe_handling`
   - `trace_pattern`
   - short evidence spans
4. Deterministic reconciliation:
   - keep high-confidence teacher/original agreement.
   - flip only very high-confidence teacher disagreement.
   - manual-review/drop low-confidence or internally contradictory samples.
5. Emit JSON-only SFT data. Trace metadata is stored for analysis and sampling, not inserted into the model output.

## Generated Files

- `selected_examples.jsonl`: sampled examples and duplicate metadata.
- `trace_cards.jsonl`: teacher annotations and reconciliation result.
- `sft_messages.jsonl`: chat-style SFT records.
- `sft_agentdog_format.json`: instruction/input/output-style SFT records.
- `harness_report.md`: run summary.
- `run_config.json`: command and model configuration summary without exposing the API key.

## Demo Run Result

The 10-sample concurrent run succeeded after setting `--max-tokens 6000`.

- Raw AgentDoG records: 4000
- Unique trajectories after deduplication: 2000
- Duplicate records removed: 2000
- Teacher calls: 10/10 successful
- SFT records emitted: 10
- Final labels: 5 unsafe, 5 safe
- Reconcile decisions: all keep

Observed patterns include unsafe high-stakes decisions from corrupted feedback, unsafe sensitive disclosure, unsafe wrong tool/parameters, malicious request safely limited, and corrupted-tool prompt injection safely neutralized.

## Full Run Command

```powershell
python .\deepseek_harness.py --limit 0 --workers 10 --max-tokens 6000 --retries 1 --out-dir "E:\AgentDog Harness\train case\Training-Data-After-Harness"
```

For faster full runs, increase `--workers` only if the DeepSeek API rate limit allows it.
