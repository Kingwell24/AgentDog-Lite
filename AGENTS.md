# AGENTS.md

## Project Rule: Local Machine vs GPU Server

The local machine is strictly for code editing, documentation, formatting, and lightweight validation.

Do not run real model inference, model downloads, fine-tuning, benchmark sweeps, or any GPU-heavy workload on the local machine. All `Qwen/Qwen3.5-0.8B` baseline runs, LoRA/QLoRA training, and evaluation jobs must be executed on the GPU server.

Scripts in this repository should preserve this rule by default. If a script has an explicit local override flag, use it only when a human intentionally asks for local execution.

## Hackathon Defaults

- Keep changes small and reviewable.
- Prefer the shortest runnable baseline before adding training or optimization.
- Do not invent metrics or results. If a run has not been executed on the server, report it as not yet run.
- Keep baseline outputs under `outputs/` and model weights/adapters under `models/` or `adapters/`; do not commit generated weights or large result artifacts unless the team explicitly decides to.

## Training Checkpoint and Result Retention

- Every new training run must write to a fresh run folder instead of overwriting an earlier run.
- Keep all useful checkpoints for that run, including the model-weight checkpoint when produced and the LoRA adapter checkpoint.
- Save evaluation outputs for the same run in the corresponding folder, including predictions, metrics, config, and logs when available.
- Include the evaluated `acc` and `f1` in the saved checkpoint or run folder name after evaluation, so the best record can be found without opening every metrics file.
- Use a stable naming pattern such as `outputs/train_runs/<run_id>__acc-0.742__f1-0.681/` and mirror final adapters under `adapters/<run_id>__acc-0.742__f1-0.681/`.
