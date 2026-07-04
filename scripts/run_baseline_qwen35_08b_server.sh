#!/usr/bin/env bash
set -euo pipefail

# Run this on the GPU server, not on the local coding machine.

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
LIMIT_ARGS=()
if [[ "${LIMIT:-}" != "" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
BATCH_ARGS=()
if [[ "${BATCH_SIZE:-}" != "" ]]; then
  BATCH_ARGS=(--batch-size "$BATCH_SIZE")
fi

python scripts/run_baseline_qwen35_08b.py \
  --model "$MODEL" \
  --input "test case/summer_camp_ATBench300.json" \
  --output-dir outputs/baseline_qwen35_08b/atbench300 \
  "${LIMIT_ARGS[@]}" \
  "${BATCH_ARGS[@]}"

python scripts/run_baseline_qwen35_08b.py \
  --model "$MODEL" \
  --input "test case/summer_camp_rjudge.json" \
  --output-dir outputs/baseline_qwen35_08b/rjudge \
  "${LIMIT_ARGS[@]}" \
  "${BATCH_ARGS[@]}"
