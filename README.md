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
