#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}"

if [ ! -x "${LLAMA_SERVER_BIN}" ]; then
    echo "Missing llama-server: ${LLAMA_SERVER_BIN}"
    exit 1
fi
command -v curl >/dev/null
command -v numactl >/dev/null

pid_file="${RUN_DIR}/fp32.pid"
log_file="${RUN_DIR}/fp32.log"
restart="${RESTART:-0}"

print_port_owner() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :${port}" || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN || true
    fi
}

is_port_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :${port}" | awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }'
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    else
        return 1
    fi
}

if [ -f "${pid_file}" ] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    if [ "${restart}" != "1" ]; then
        echo "FP32 server is already running: pid=$(cat "${pid_file}") port=${FP32_PORT}"
        echo "Log: ${log_file}"
        exit 0
    fi
    kill "$(cat "${pid_file}")" 2>/dev/null || true
    sleep 1
fi

if is_port_listening "${FP32_PORT}"; then
    echo "FP32 port ${FP32_PORT} is already in use. Stop that process or set FP32_PORT to another port."
    print_port_owner "${FP32_PORT}"
    exit 1
fi

nohup numactl \
    --physcpubind="${FP32_CPU_BIND}" \
    --membind="${FP32_NUMA_MEMBIND}" \
    env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_LOCAL}" \
    GGML_FUSED_CPP_SDPA=0 \
    OMP_NUM_THREADS="${FP32_THREADS}" \
    OMP_DYNAMIC=FALSE \
    OPENBLAS_NUM_THREADS="${FP32_THREADS}" \
    MKL_NUM_THREADS="${FP32_THREADS}" \
    "${LLAMA_SERVER_BIN}" \
        -m "${FP32_MODEL}" \
        --embedding \
        --pooling mean \
        --embd-normalize 2 \
        --host 127.0.0.1 \
        --port "${FP32_PORT}" \
        --ctx-size "${FP32_CONTEXT_SIZE}" \
        --flash-attn "${FP32_FLASH_ATTN}" \
        -t "${FP32_THREADS}" \
        -tb "${FP32_THREADS_BATCH}" \
        -np "${FP32_PARALLEL}" \
        -b "${FP32_BATCH_TOKENS}" \
        -ub "${FP32_UBATCH_TOKENS}" \
    > "${log_file}" 2>&1 &

echo "$!" > "${pid_file}"

for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${FP32_PORT}/v1/models" >/dev/null 2>&1; then
        echo "FP32 baseline server started: pid=$(cat "${pid_file}") port=${FP32_PORT}"
        echo "Log: ${log_file}"
        exit 0
    fi
    if ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
        echo "FP32 server exited during startup. Log:"
        tail -n 80 "${log_file}" || true
        exit 1
    fi
    sleep 1
done

echo "FP32 server pid=$(cat "${pid_file}") port=${FP32_PORT}; health check did not become ready in time."
echo "Log: ${log_file}"
