# MTEB accuracy comparison for llama.cpp embedding

This directory contains local scripts for running MTEB against two llama.cpp
embedding servers on the current machine.

Default assumptions:

- build dir: `build-kleidiai-clang22`
- HF cache root: `/mnt/models/hf`
- FP32 model: `/mnt/models/bge-small-zh-v1.5-F32.gguf`
- Q8 model: `/mnt/models/bge-small-zh-v1.5-Q8_0.gguf`
- FP32 slot context: `512`, total context: `8192`
- Q8 slot context: `512`, total context: `512`
- FP32 CPU bind: `80-239`, NUMA memory bind: `1,2`, threads: `160`
- FP32 batching: `-np 16`, `-b 32768`, `-ub 2048`, MTEB batch size `64`
- Q8 CPU bind: `240`, NUMA memory bind: `3`, threads: `1`

Scripts:

- `start_fp32_original_attention.sh`: starts FP32 baseline server with fused
  attention disabled. It uses multi-threading and request batching by default.
- `start_q8_fused_attention_single_thread.sh`: starts Q8 server with fused_cpp
  attention enabled and strict single-thread defaults.
- `test_fp32_mteb.sh`: runs MTEB against the FP32 server.
- `test_q8_mteb.sh`: runs MTEB against the Q8 server.

Quick start:

```bash
cd /home/zhangxu/codex/llama.cpp-origin-master/mteb_accuracy_compare

./start_fp32_original_attention.sh
./test_fp32_mteb.sh

./start_q8_fused_attention_single_thread.sh
./test_q8_mteb.sh
```

Useful overrides:

```bash
MTEB_TASKS="MIRACLReranking" MTEB_LANGUAGES="zho" ./test_fp32_mteb.sh
MTEB_TASKS="MIRACLReranking" MTEB_LANGUAGES="zho" ./test_q8_mteb.sh
```

By default the test scripts run in quick mode with `MTEB_MAX_EVAL_SAMPLES=500`.
For full runs:

```bash
MTEB_QUICK=0 ./test_fp32_mteb.sh
MTEB_QUICK=0 ./test_q8_mteb.sh
```

The selected Python environment must have `mteb`, `datasets`, `numpy`,
`requests`, and `tqdm` installed. Missing dependencies, missing server binary,
or a stopped server fail directly.

Use `RESTART=1` when changing CPU or NUMA binding for an already running
server:

```bash
RESTART=1 ./start_fp32_original_attention.sh
RESTART=1 ./start_q8_fused_attention_single_thread.sh
```

If the port is already occupied, stop the old process or override the port:

```bash
FP32_PORT=18082 ./start_fp32_original_attention.sh
Q8_PORT=18083 ./start_q8_fused_attention_single_thread.sh
```
