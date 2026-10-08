#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to a trained ChemDiffAgent checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the downloaded dataset data directory}"

export MODEL_PATH DATA_DIR
export OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/outputs/inference"}
export CONDA_ENV=${CONDA_ENV:-chemdiffagent}
export BLOCK_LENGTH=${BLOCK_LENGTH:-64}
export DENOISING_STEPS=${DENOISING_STEPS:-64}
export MAX_TURNS=${MAX_TURNS:-7}

bash "${ROOT}/inference/run_dllm_inference.sh" "$@"

