#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct Shape {
    int64_t B;
    int64_t H;
    int64_t L;
    int64_t S;
    int64_t D;
    int64_t Dv;
};

enum class MaskKind {
    None,
    F16,
    F32,
};

static size_t idx_qkv(
        int64_t d, int64_t t, int64_t h, int64_t b,
        int64_t d_size, int64_t t_size, const Shape & shape) {
    return static_cast<size_t>(((b*shape.H + h)*t_size + t)*d_size + d);
}

static size_t idx_out(int64_t d, int64_t h, int64_t l, int64_t b, const Shape & shape) {
    return static_cast<size_t>(((b*shape.L + l)*shape.H + h)*shape.Dv + d);
}

static size_t idx_mask(int64_t s, int64_t l, int64_t b, const Shape & shape) {
    return static_cast<size_t>((b*shape.L + l)*shape.S + s);
}

static void fill_inputs(
        std::vector<float> & q,
        std::vector<float> & k,
        std::vector<float> & v) {
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = std::sin(0.13f*static_cast<float>(i + 1));
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = std::cos(0.07f*static_cast<float>(i + 3));
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = std::sin(0.11f*static_cast<float>(i + 5))*0.5f;
    }
}

static void fill_mask(std::vector<float> & mask, const Shape & shape) {
    for (int64_t b = 0; b < shape.B; ++b) {
        for (int64_t l = 0; l < shape.L; ++l) {
            for (int64_t s = 0; s < shape.S; ++s) {
                const float v = s > l ? -10000.0f : 0.0f;
                mask[idx_mask(s, l, b, shape)] = v;
            }
        }
    }
}

static void reference_sdpa(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> * mask,
        std::vector<float> & out,
        float scale,
        const Shape & shape) {
    std::vector<float> scores(static_cast<size_t>(shape.S));

    for (int64_t b = 0; b < shape.B; ++b) {
        for (int64_t h = 0; h < shape.H; ++h) {
            for (int64_t l = 0; l < shape.L; ++l) {
                float max_score = -std::numeric_limits<float>::infinity();

                for (int64_t s = 0; s < shape.S; ++s) {
                    float score = 0.0f;
                    for (int64_t d = 0; d < shape.D; ++d) {
                        score += q[idx_qkv(d, l, h, b, shape.D, shape.L, shape)]*
                                 k[idx_qkv(d, s, h, b, shape.D, shape.S, shape)];
                    }
                    score *= scale;
                    if (mask != nullptr) {
                        score += (*mask)[idx_mask(s, l, b, shape)];
                    }
                    scores[static_cast<size_t>(s)] = score;
                    max_score = std::max(max_score, score);
                }

                float denom = 0.0f;
                std::fill(
                        out.begin() + idx_out(0, h, l, b, shape),
                        out.begin() + idx_out(0, h, l, b, shape) + shape.Dv,
                        0.0f);

                for (int64_t s = 0; s < shape.S; ++s) {
                    const float prob = std::exp(scores[static_cast<size_t>(s)] - max_score);
                    denom += prob;
                    for (int64_t d = 0; d < shape.Dv; ++d) {
                        out[idx_out(d, h, l, b, shape)] +=
                            prob*v[idx_qkv(d, s, h, b, shape.Dv, shape.S, shape)];
                    }
                }

                const float inv_denom = 1.0f/denom;
                for (int64_t d = 0; d < shape.Dv; ++d) {
                    out[idx_out(d, h, l, b, shape)] *= inv_denom;
                }
            }
        }
    }
}

static bool run_case(const Shape & shape, MaskKind mask_kind) {
    ggml_init_params params = {
        /* .mem_size   = */ 32u*1024u*1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape.D,  shape.L, shape.H, shape.B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape.D,  shape.S, shape.H, shape.B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape.Dv, shape.S, shape.H, shape.B);

    ggml_tensor * mask_tensor = nullptr;
    std::vector<float> mask_ref(static_cast<size_t>(shape.B*shape.L*shape.S));
    fill_mask(mask_ref, shape);
    std::vector<ggml_fp16_t> mask_f16(mask_ref.size());
    if (mask_kind == MaskKind::F16) {
        mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, shape.S, shape.L, 1, shape.B);
        for (size_t i = 0; i < mask_ref.size(); ++i) {
            mask_f16[i] = ggml_fp32_to_fp16(mask_ref[i]);
        }
    } else if (mask_kind == MaskKind::F32) {
        mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape.S, shape.L, 1, shape.B);
    }

    const float scale = 1.0f/std::sqrt(static_cast<float>(shape.D));
    ggml_tensor * out = ggml_fused_cpp_sdpa_ext(ctx, q, k, v, mask_tensor, scale);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    std::vector<float> q_data(static_cast<size_t>(shape.B*shape.H*shape.L*shape.D));
    std::vector<float> k_data(static_cast<size_t>(shape.B*shape.H*shape.S*shape.D));
    std::vector<float> v_data(static_cast<size_t>(shape.B*shape.H*shape.S*shape.Dv));
    fill_inputs(q_data, k_data, v_data);

    std::memcpy(q->data, q_data.data(), q_data.size()*sizeof(float));
    std::memcpy(k->data, k_data.data(), k_data.size()*sizeof(float));
    std::memcpy(v->data, v_data.data(), v_data.size()*sizeof(float));
    if (mask_kind == MaskKind::F16) {
        std::memcpy(mask_tensor->data, mask_f16.data(), mask_f16.size()*sizeof(ggml_fp16_t));
    } else if (mask_kind == MaskKind::F32) {
        std::memcpy(mask_tensor->data, mask_ref.data(), mask_ref.size()*sizeof(float));
    }

    ggml_cplan cplan = ggml_graph_plan(gf, 1, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size);
    cplan.work_data = work_data.data();

    if (ggml_graph_compute(gf, &cplan) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<float> ref(static_cast<size_t>(shape.B*shape.L*shape.H*shape.Dv));
    reference_sdpa(q_data, k_data, v_data, mask_kind == MaskKind::None ? nullptr : &mask_ref, ref, scale, shape);

    const float * got = static_cast<const float *>(out->data);
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        if (diff > 1e-4f) {
            std::fprintf(stderr,
                    "mismatch B=%lld H=%lld L=%lld S=%lld D=%lld Dv=%lld mask=%d at %zu: got=%g ref=%g diff=%g\n",
                    (long long) shape.B, (long long) shape.H,
                    (long long) shape.L, (long long) shape.S,
                    (long long) shape.D, (long long) shape.Dv,
                    static_cast<int>(mask_kind), i, got[i], ref[i], diff);
            ggml_free(ctx);
            return false;
        }
    }

    ggml_free(ctx);
    return true;
}

} // namespace

int main() {
#if defined(__aarch64__) || defined(_M_ARM64)
    const Shape shapes[] = {
        {1, 2,  8,  8, 16, 16},
        {1, 2, 10, 10, 16, 16},
        {1, 2, 13, 13, 16, 16},
        {1, 2, 15, 15, 16, 16},
        {1, 8, 10, 10, 64, 64},
    };

    for (const Shape & shape : shapes) {
        for (MaskKind mask_kind : {MaskKind::None, MaskKind::F16, MaskKind::F32}) {
            if (!run_case(shape, mask_kind)) {
                return 1;
            }
        }
    }
    return 0;
#else
    std::printf("fused_cpp SDPA test skipped on non-ARM platform\n");
    return 0;
#endif
}
