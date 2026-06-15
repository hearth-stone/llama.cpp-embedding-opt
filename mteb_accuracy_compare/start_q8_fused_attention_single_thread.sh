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

pid_file="${RUN_DIR}/q8.pid"
log_file="${RUN_DIR}/q8.log"
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
        echo "Q8 fused server is already running: pid=$(cat "${pid_file}") port=${Q8_PORT}"
        echo "Log: ${log_file}"
        exit 0
    fi
    kill "$(cat "${pid_file}")" 2>/dev/null || true
    sleep 1
fi

if is_port_listening "${Q8_PORT}"; then
    echo "Q8 port ${Q8_PORT} is already in use. Stop that process or set Q8_PORT to another port."
    print_port_owner "${Q8_PORT}"
    exit 1
fi

nohup numactl \
    --physcpubind="${Q8_CPU_BIND}" \
    --membind="${Q8_NUMA_MEMBIND}" \
    env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_LOCAL}" \
    GGML_FUSED_CPP_SDPA="${Q8_FUSED_CPP_SDPA}" \
    GGML_TOTAL_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "${LLAMA_SERVER_BIN}" \
        -m "${Q8_MODEL}" \
        --embedding \
        --pooling mean \
        --embd-normalize 2 \
        --host 127.0.0.1 \
        --port "${Q8_PORT}" \
        --ctx-size "${Q8_CONTEXT_SIZE}" \
        --flash-attn on \
        -t 1 \
        -tb 1 \
        -np 1 \
        -b "${Q8_BATCH_TOKENS}" \
        -ub "${Q8_UBATCH_TOKENS}" \
    > "${log_file}" 2>&1 &

echo "$!" > "${pid_file}"

for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${Q8_PORT}/v1/models" >/dev/null 2>&1; then
        echo "Q8 fused server started: pid=$(cat "${pid_file}") port=${Q8_PORT}"
        echo "Log: ${log_file}"
        exit 0
    fi
    if ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
        echo "Q8 server exited during startup. Log:"
        tail -n 80 "${log_file}" || true
        exit 1
    fi
    sleep 1
done

echo "Q8 server pid=$(cat "${pid_file}") port=${Q8_PORT}; health check did not become ready in time."
echo "Log: ${log_file}"
