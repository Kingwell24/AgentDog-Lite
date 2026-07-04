# AgentDoG-Lite Baseline

This repository is for the AgentDoG-Lite hackathon task: judge whether an agent trajectory is `safe` or `unsafe`, starting from the base model `Qwen/Qwen3.5-0.8B`.

Strict rule: this local machine is only for code editing and lightweight validation. Run real model inference, GPU use, model downloads, and fine-tuning on the GPU server.

## Baseline

The baseline script reads the provided JSON test set, formats each trajectory and tool list, prompts `Qwen/Qwen3.5-0.8B`, parses `{"judgment":"safe|unsafe"}`, and writes predictions plus metrics.

Install on the GPU server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U torch transformers accelerate
```

Run a small smoke baseline on the server:

```bash
LIMIT=5 BATCH_SIZE=8 bash scripts/run_baseline_qwen35_08b_server.sh
```

Run the full baseline on the server:

```bash
BATCH_SIZE=16 bash scripts/run_baseline_qwen35_08b_server.sh
```

Outputs:

- `predictions.jsonl`: `id`, gold `label`, parsed `prediction`, raw model output, input/output token counts.
- `metrics.json`: accuracy, precision, recall, F1, invalid output rate, average output tokens, confusion matrix.
- `run_config.json`: model, input, prompt, and generation settings.

## LLaMA-Factory LoRA SFT

This flow is command-line only. Do not use the LLaMA-Factory Web UI. Run the data download, training, adapter inference, and evaluation on the GPU server.

Install training dependencies on the GPU server:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
python -m pip install -e ".[torch,metrics]"

cd /path/to/AgentDoG-Lite
python -m pip install -U datasets transformers peft accelerate bitsandbytes
```

Prepare the mixed AgentDoG1.0 SFT file:

```bash
python scripts/prepare_agentdog_data.py \
  --output data/agentdog_mix_sft.jsonl \
  --stats-output data/agentdog_mix_sft_stats.json \
  --seed 42
```

The converter reads `AI45Research/AgentDoG1.0-Training-Data`, converts `AgentDoG-BinarySafety` to short `{"judgment":"safe|unsafe"}` targets, converts `AgentDoG-FineGrainedTaxonomy` to unsafe diagnostic JSON targets, and oversamples BinarySafety safe examples so the final safe/unsafe mix is roughly balanced.

Start LoRA training:

```bash
bash scripts/run_llamafactory_train.sh configs/qwen35_08b_lora_sft.yaml
```

Use QLoRA only if the LoRA run hits memory pressure:

```bash
bash scripts/run_llamafactory_train.sh configs/qwen35_08b_qlora_sft.yaml
```

Each training run writes a fresh directory under:

```text
outputs/train_runs/<run_id>__acc-pending__f1-pending/
```

The run directory keeps `source_config.yaml`, the resolved `train_config.yaml`, `train.log`, LLaMA-Factory checkpoints, adapter files, and later `eval/` outputs. Do not overwrite earlier run folders.

Run adapter inference on a test set:

```bash
RUN_DIR=outputs/train_runs/<run_id>__acc-pending__f1-pending

python scripts/run_inference.py \
  --base-model Qwen/Qwen3.5-0.8B \
  --adapter "$RUN_DIR" \
  --input "test case/summer_camp_ATBench300.json" \
  --output "$RUN_DIR/eval/atbench300/predictions.jsonl" \
  --batch-size 16
```

Evaluate and tag the run by score:

```bash
python scripts/evaluate.py \
  --predictions "$RUN_DIR/eval/atbench300/predictions.jsonl" \
  --metrics-output "$RUN_DIR/eval/metrics.json" \
  --tag-run-dir "$RUN_DIR" \
  --adapter-output-root adapters
```

After evaluation, the run folder is renamed to include `acc` and `f1`, for example:

```text
outputs/train_runs/qwen35_08b_lora_sft-20260704-213000__acc-0.742__f1-0.681/
adapters/qwen35_08b_lora_sft-20260704-213000__acc-0.742__f1-0.681/
```

Local lightweight checks that do not run model inference:

```bash
python scripts/prepare_agentdog_data.py --self-test
python scripts/evaluate.py --self-test
python scripts/test_llamafactory_flow.py
python scripts/test_baseline_qwen35_08b.py
```

## Generic HF Test Script

For the released summer camp test set, run this on the GPU server:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/run_qwen35_08b_infer_test.py \
  --source hf \
  --hf-repo AI45Research/2026_summer_camp_teseset \
  --limit 1
```

For local files already copied into `test case/`:

```bash
python scripts/run_qwen35_08b_infer_test.py \
  --source local \
  --input "test case/summer_camp_ATBench300.json" \
  --input "test case/summer_camp_rjudge.json" \
  --limit 1 \
  --batch-size 8
```

For throughput runs on the A800 server, try `--batch-size 16` first, then compare
`--batch-size 32` if memory remains low.

Per-sample debug details, including `raw_output`, are saved in each output
directory's `predictions.jsonl`. Use `--print-raw` only when you intentionally
want raw model output in the terminal.
