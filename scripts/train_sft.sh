#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${BASE_MODEL:?Set BASE_MODEL to the SDAR-8B-Chat path or Hub ID}"
: "${SFT_DATA:?Set SFT_DATA to the downloaded training/train.json}"
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/outputs/chemdiffagent-sft"}
MASTER_PORT=${MASTER_PORT:-29517}

cd "${ROOT}/training/sft"
accelerate launch \
  --config_file accelerate_configs/8_gpus_zero3.yaml \
  --main_process_port "${MASTER_PORT}" \
  train/sft_sdar.py \
  config=configs/chemdiffagent_sft.yaml \
  model.pretrained_model="${BASE_MODEL}" \
  dataset.path="${SFT_DATA}" \
  experiment.output_dir="${OUTPUT_DIR}" \
  "$@"
