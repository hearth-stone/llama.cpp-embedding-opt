#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

timestamp="$(date +%Y%m%d-%H%M%S)"
output_folder="${Q8_MTEB_OUTPUT:-${RUN_DIR}/results/q8-fused-${timestamp}}"
mkdir -p "${RUN_DIR}" "$(dirname "${output_folder}")"

"${PYTHON}" -c "import mteb, datasets, numpy, requests, tqdm"
command -v curl >/dev/null

if ! curl -fsS "http://127.0.0.1:${Q8_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Q8 fused server is not healthy on port ${Q8_PORT}. Start it first with start_q8_fused_attention_single_thread.sh."
    exit 1
fi

export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export HF_HUB_OFFLINE="${HF_OFFLINE}"
export HF_DATASETS_OFFLINE="${HF_OFFLINE}"

cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/run_mteb_llamacpp.py"
    --base-url "http://127.0.0.1:${Q8_PORT}"
    --model "${MTEB_MODEL_NAME}"
    --batch-size "${Q8_MTEB_BATCH_SIZE}"
    --output-folder "${output_folder}"
    --max-tokens "${MTEB_MAX_TOKENS}"
    --timeout "${MTEB_TIMEOUT}"
)

if [ -n "${MTEB_TASKS}" ]; then
    cmd+=(--tasks)
    for task in ${MTEB_TASKS}; do
        cmd+=("${task}")
    done
fi

if [ -n "${MTEB_LANGUAGES}" ]; then
    cmd+=(--languages)
    for language in ${MTEB_LANGUAGES}; do
        cmd+=("${language}")
    done
fi

if [ "${MTEB_QUICK}" = "1" ]; then
    cmd+=(--quick)
fi

if [ -n "${MTEB_MAX_EVAL_SAMPLES}" ]; then
    cmd+=(--max-eval-samples "${MTEB_MAX_EVAL_SAMPLES}")
fi
if [ -n "${MTEB_SAMPLES_PER_LABEL}" ]; then
    cmd+=(--samples-per-label "${MTEB_SAMPLES_PER_LABEL}")
fi
if [ -n "${MTEB_N_EXPERIMENTS}" ]; then
    cmd+=(--n-experiments "${MTEB_N_EXPERIMENTS}")
fi
if [ "${MTEB_NO_BGE_QUERY_INSTRUCTION}" = "1" ]; then
    cmd+=(--no-bge-query-instruction)
fi

printf 'Running Q8 fused MTEB:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
echo "Q8 fused MTEB results: ${output_folder}"
