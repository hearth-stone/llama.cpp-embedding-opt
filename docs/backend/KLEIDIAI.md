# Build with Arm KleidiAI v1.24.0

This guide builds llama.cpp with the CPU KleidiAI backend pinned to
KleidiAI `v1.24.0`.

KleidiAI is only useful on Arm CPUs. It accelerates selected CPU tensor
operations when the model, tensor layout, quantization type, and runtime CPU
features match the supported kernels.

## Prerequisites

- An AArch64 machine or an Arm cross-compilation environment.
- CMake and a C/C++ compiler supported by llama.cpp.
- Network access to GitHub, unless you provide the KleidiAI source tree
  locally.

The llama.cpp CMake configuration already pins KleidiAI to `v1.24.0`:

```cmake
set(KLEIDIAI_COMMIT_TAG "v1.24.0")
set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/releases/download/${KLEIDIAI_COMMIT_TAG}/kleidiai-${KLEIDIAI_COMMIT_TAG}-src.tar.gz")
set(KLEIDIAI_RELEASE_ARCHIVE_MD5 "2f02ebe29573d45813e671eb304f2a00")
```

## Online build

From the llama.cpp source directory:

```bash
cmake -S . -B build-kleidiai \
  -DGGML_NATIVE=ON \
  -DGGML_CPU_KLEIDIAI=ON

cmake --build build-kleidiai --config Release -j
```

With `GGML_CPU_KLEIDIAI=ON`, CMake downloads and verifies the
`kleidiai-v1.24.0-src.tar.gz` archive automatically.

If your environment does not have `ccache`, or if `ccache` cannot write to its
cache directory, add:

```bash
-DGGML_CCACHE=OFF
```

## Local pinned source build

Use this path when the build machine cannot download during CMake
configuration, or when you want every builder to use the same local KleidiAI
source tree.

```bash
mkdir -p third_party
git clone --depth 1 --branch v1.24.0 \
  https://github.com/ARM-software/kleidiai.git \
  third_party/kleidiai-v1.24.0

cmake -S . -B build-kleidiai \
  -DGGML_NATIVE=ON \
  -DGGML_CPU_KLEIDIAI=ON \
  -DFETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD="$PWD/third_party/kleidiai-v1.24.0"

cmake --build build-kleidiai --config Release -j
```

`FETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD` must point at the KleidiAI source
root that contains the `kai/` directory.

## Local archive build

If you prefer to distribute the official source archive:

```bash
mkdir -p third_party/kleidiai-v1.24.0
curl -L \
  -o third_party/kleidiai-v1.24.0-src.tar.gz \
  https://github.com/ARM-software/kleidiai/releases/download/v1.24.0/kleidiai-v1.24.0-src.tar.gz

cmake -E md5sum third_party/kleidiai-v1.24.0-src.tar.gz
```

The checksum must be:

```text
2f02ebe29573d45813e671eb304f2a00
```

Extract it and build:

```bash
tar -xzf third_party/kleidiai-v1.24.0-src.tar.gz \
  -C third_party/kleidiai-v1.24.0 \
  --strip-components=1

cmake -S . -B build-kleidiai \
  -DGGML_NATIVE=ON \
  -DGGML_CPU_KLEIDIAI=ON \
  -DFETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD="$PWD/third_party/kleidiai-v1.24.0"

cmake --build build-kleidiai --config Release -j
```

## Verify the build

Run a CPU path to confirm that tensors are loaded into the KleidiAI CPU buffer:

```bash
./build-kleidiai/bin/llama-cli \
  -m /path/to/model.gguf \
  -p "What is a car?" \
  --device none
```

A successful KleidiAI load prints a line similar to:

```text
load_tensors: CPU_KLEIDIAI model buffer size =  3474.00 MiB
```

If higher-priority backends are enabled, such as Metal on macOS, tensors may be
placed on those backends instead. For CPU-only verification, disable them at
configure time, for example:

```bash
cmake -S . -B build-kleidiai \
  -DGGML_NATIVE=ON \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_METAL=OFF
```

or force CPU execution at runtime with:

```bash
--device none
```

## Runtime controls

KleidiAI selects kernels at runtime based on CPU features such as dotprod,
i8mm, SVE, and SME.

SME behavior can be controlled with `GGML_KLEIDIAI_SME`:

- Not set: detect SME support automatically.
- `0`: disable SME kernels.
- `<n>`: enable SME kernels and assume `<n>` available SME units.

The KleidiAI chunk multiplier can be controlled with
`GGML_KLEIDIAI_CHUNK_MULTIPLIER` when tuning thread partitioning.

The fused_cpp embedding SDPA path is separate from KleidiAI matmul kernels. It
is enabled with `GGML_FUSED_CPP_SDPA=1` and, by default, uses the NEON QKT/PV
microkernels. The default path skips the online-softmax correction rescale when
there is no previous S-block contribution, because the running accumulator is
still zero.

`FUSED_CPP_SDPA_QKT_ROWMAX=1` enables the experimental NEON QKT row-max fusion.
It updates the per-row score maximum while QKT writes the score tile, then lets
softmax skip its separate max scan. On Arm-codex, with single-core
`B=1,H=8,L=512,S=512,D=64,DV=64` and an all-zero f16 mask, this measured
9.71 ms, roughly the same as the default path after correction-rescale skip.
Keep it opt-in because the gain is small and shape-dependent.

For local kernel experiments, `FUSED_CPP_SDPA_USE_SVE_KERNELS=1` selects the
SVE QKT/PV microkernels when the binary is built on AArch64 with SVE compiler
support. This is opt-in because it is not faster on every SVE width. On
Arm-codex with 256-bit SVE, single-core Q8 embedding at input length 512 was
faster with the default NEON path than with the SVE path, so leave this unset
unless explicitly profiling the SVE kernels.

`FUSED_CPP_SDPA_USE_SVE_QKT_4X32=1` additionally enables the experimental SVE
QKT 4x32 kernel. This is only for profiling; on Arm-codex it was slower than
the default SVE QKT path because it reloads each K tile for two 4-row halves.

## Common issues

If CMake still downloads KleidiAI after you provided a local source tree,
delete the build directory and configure again. FetchContent caches source
state inside the build directory.

If `CPU_KLEIDIAI` does not appear in the load log, check that:

- The project was configured with `-DGGML_CPU_KLEIDIAI=ON`.
- The model uses tensor types supported by the KleidiAI integration.
- The CPU supports at least one kernel path selected by llama.cpp.
- No higher-priority backend took the tensors.

If the build fails in downloaded KleidiAI sources, confirm that every builder is
using `v1.24.0` and not a moving local branch.
