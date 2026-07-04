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
LIMIT=5 bash scripts/run_baseline_qwen35_08b_server.sh
```

Run the full baseline on the server:

```bash
bash scripts/run_baseline_qwen35_08b_server.sh
```

Outputs:

- `predictions.jsonl`: `id`, gold `label`, parsed `prediction`, raw model output, input/output token counts.
- `metrics.json`: accuracy, precision, recall, F1, invalid output rate, average output tokens, confusion matrix.
- `run_config.json`: model, input, prompt, and generation settings.
