#!/usr/bin/env bash
set -euo pipefail

# Run this on the GPU server, not on the local coding machine.

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_llamafactory_train.sh configs/qwen35_08b_lora_sft.yaml" >&2
  exit 2
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli not found. Install LLaMA-Factory on the GPU server first." >&2
  exit 127
fi

CONFIG="$1"
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-outputs/train_runs}"
CONFIG_STEM="$(basename "$CONFIG" .yaml)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-${CONFIG_STEM}-${TIMESTAMP}}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}__acc-pending__f1-pending"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/eval" "$RUN_DIR/logs"
TRAIN_CONFIG="$RUN_DIR/train_config.yaml"
SOURCE_CONFIG="$RUN_DIR/source_config.yaml"
RUN_META="$RUN_DIR/run_metadata.json"

cp "$CONFIG" "$SOURCE_CONFIG"
sed "s|__OUTPUT_DIR__|$RUN_DIR|g" "$CONFIG" > "$TRAIN_CONFIG"

cat > "$RUN_META" <<EOF
{
  "run_name": "$RUN_NAME",
  "run_dir": "$RUN_DIR",
  "source_config": "$CONFIG",
  "train_config": "$TRAIN_CONFIG",
  "created_at": "$TIMESTAMP"
}
EOF

echo "Run directory: $RUN_DIR"
echo "Training config: $TRAIN_CONFIG"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
llamafactory-cli train "$TRAIN_CONFIG" 2>&1 | tee "$RUN_DIR/train.log"
