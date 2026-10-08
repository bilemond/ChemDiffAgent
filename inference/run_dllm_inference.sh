#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CONDA_ENV=${CONDA_ENV:-chemdiffagent}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES
export RXN_BACKEND=${RXN_BACKEND:-t5chem}
export RXN_T5CHEM_ROOT=${RXN_T5CHEM_ROOT:-${PROJECT_ROOT}/third_party/t5chem}
export RXN_T5CHEM_DEVICE=${RXN_T5CHEM_DEVICE:-cuda:0}
export UNICORE_ROOT=${UNICORE_ROOT:-${PROJECT_ROOT}/third_party/Uni-Core}

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a trained checkpoint}
DATA_DIR=${DATA_DIR:?Set DATA_DIR to the downloaded dataset data directory}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/inference}
DATASETS=${DATASETS:-single_test,multi_test}

MAX_SAMPLES=${MAX_SAMPLES:-0}
SAMPLE_ID=${SAMPLE_ID:-}
NUM_ROLLS=${NUM_ROLLS:-1}
MAX_TURNS=${MAX_TURNS:-7}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-32768}
BLOCK_LENGTH=${BLOCK_LENGTH:-64}
DENOISING_STEPS=${DENOISING_STEPS:-64}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-1}
REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_static}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.9}
DTYPE=${DTYPE:-bf16}
TOOL_TIMEOUT_SECONDS=${TOOL_TIMEOUT_SECONDS:-120}
RESUME=${RESUME:-0}
PRINT_TURNS=${PRINT_TURNS:-0}

mkdir -p "${OUTPUT_DIR}"
IFS=',' read -r -a DATASET_NAMES <<< "${DATASETS}"

for dataset in "${DATASET_NAMES[@]}"; do
    input_path="${DATA_DIR}/${dataset}.jsonl"
    output_path="${OUTPUT_DIR}/${dataset}.jsonl"
    if [[ ! -f "${input_path}" ]]; then
        echo "Missing dataset, skipping: ${input_path}" >&2
        continue
    fi

    args=(
        --input "${input_path}"
        --output "${output_path}"
        --model-path "${MODEL_PATH}"
        --dtype "${DTYPE}"
        --max-samples "${MAX_SAMPLES}"
        --num-rolls "${NUM_ROLLS}"
        --max-turns "${MAX_TURNS}"
        --max-new-tokens "${MAX_NEW_TOKENS}"
        --max-context-tokens "${MAX_CONTEXT_TOKENS}"
        --block-length "${BLOCK_LENGTH}"
        --denoising-steps "${DENOISING_STEPS}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --top-k "${TOP_K}"
        --remasking-strategy "${REMASKING_STRATEGY}"
        --confidence-threshold "${CONFIDENCE_THRESHOLD}"
        --tool-timeout-seconds "${TOOL_TIMEOUT_SECONDS}"
    )
    [[ -n "${SAMPLE_ID}" ]] && args+=(--sample-id "${SAMPLE_ID}")
    [[ "${RESUME}" != "0" ]] && args+=(--resume)
    [[ "${PRINT_TURNS}" != "0" ]] && args+=(--print-turns)

    echo "[DLLM] dataset=${dataset} model=${MODEL_PATH} gpu=${CUDA_VISIBLE_DEVICES}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python "${SCRIPT_DIR}/dllm_inference.py" "${args[@]}" "$@"
done
