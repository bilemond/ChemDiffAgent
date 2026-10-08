#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${SFT_MODEL:?Set SFT_MODEL to the Agentic SFT checkpoint}"
: "${PREFERENCE_DATA:?Set PREFERENCE_DATA to a prompt/chosen/rejected JSONL file}"
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/outputs/chemdiffagent-vrpo"}
MASTER_PORT=${MASTER_PORT:-29518}

cd "${ROOT}/training/vrpo"
accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes 8 \
  --main_process_port "${MASTER_PORT}" \
  my_train/my_dpo_train.py \
  --config recipes/chemdiffagent_vrpo.yaml \
  --model_name_or_path "${SFT_MODEL}" \
  --dataset_path "${PREFERENCE_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

