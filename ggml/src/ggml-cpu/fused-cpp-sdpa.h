#pragma once

#include "ggml-cpu-impl.h"

struct ggml_tensor;

bool ggml_fused_cpp_sdpa_flash_attn_ext_fp32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst);
