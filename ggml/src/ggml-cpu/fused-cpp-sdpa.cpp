#include "fused-cpp-sdpa.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace {

using fused_cpp_sdpa_fn = int (*)(
        const float * q,
        const float * k,
        const float * v,
        float *       out,
        int64_t B,
        int64_t H,
        int64_t L,
        int64_t S,
        int64_t D,
        int64_t DV,
        int64_t q_nb0,
        int64_t q_nb1,
        int64_t q_nb2,
        int64_t q_nb3,
        int64_t k_nb0,
        int64_t k_nb1,
        int64_t k_nb2,
        int64_t k_nb3,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        int64_t o_nb0,
        int64_t o_nb1,
        int64_t o_nb2,
        int64_t o_nb3,
        float scale);

static bool enabled() {
    const char * env = std::getenv("GGML_FUSED_CPP_SDPA");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool debug_enabled() {
    const char * env = std::getenv("GGML_FUSED_CPP_SDPA_DEBUG");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static void debug_log(const char * msg) {
    if (debug_enabled()) {
        std::fprintf(stderr, "GGML_FUSED_CPP_SDPA: %s\n", msg);
    }
}

static const char * type_name(ggml_type type) {
    return ggml_type_name(type);
}

static float get_f32_param(const ggml_tensor * dst, int i) {
    float v = 0.0f;
    const char * p = reinterpret_cast<const char *>(dst->op_params);
    std::memcpy(&v, p + i*sizeof(float), sizeof(float));
    return v;
}

static bool supported(const ggml_compute_params * params, const ggml_tensor * dst) {
    if (!enabled()) {
        debug_log("disabled");
        return false;
    }

    if (params->use_ref) {
        debug_log("fallback: use_ref is enabled");
        return false;
    }

    if (params->nth != 1 || params->ith != 0) {
        debug_log("fallback: only nth=1, ith=0 is supported");
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    if (q == nullptr || k == nullptr || v == nullptr || dst->src[4] != nullptr) {
        debug_log("fallback: unsupported null inputs or sinks");
        return false;
    }

    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) ||
            (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16)) {
        if (debug_enabled()) {
            std::fprintf(stderr,
                    "GGML_FUSED_CPP_SDPA: fallback: unsupported q/k/v/out type, got q=%s k=%s v=%s out=%s\n",
                    type_name(q->type), type_name(k->type), type_name(v->type), type_name(dst->type));
        }
        return false;
    }

    if (get_f32_param(dst, 1) != 0.0f || get_f32_param(dst, 2) != 0.0f) {
        debug_log("fallback: max_bias or logit_softcap is not zero");
        return false;
    }

    if (q->ne[0] != k->ne[0] || k->ne[1] != v->ne[1] || q->ne[2] != k->ne[2] ||
        q->ne[2] != v->ne[2] || q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        debug_log("fallback: q/k/v shape mismatch");
        return false;
    }

    if (dst->ne[0] != v->ne[0] || dst->ne[1] != q->ne[2] ||
        dst->ne[2] != q->ne[1] || dst->ne[3] != q->ne[3]) {
        debug_log("fallback: output shape mismatch");
        return false;
    }

    if (m != nullptr) {
        if (q->ne[3] != 1 || m->ne[0] != k->ne[1] || m->ne[1] != q->ne[1] || m->ne[2] != 1 || m->ne[3] != 1) {
            debug_log("fallback: unsupported mask shape for quick validation");
            return false;
        }

        debug_log("ignoring attention mask for single-sequence quick validation");
    }

    const int64_t k_nb0 = k->type == GGML_TYPE_F32 ? (int64_t) sizeof(float) : (int64_t) sizeof(ggml_fp16_t);
    const int64_t v_nb0 = v->type == GGML_TYPE_F32 ? (int64_t) sizeof(float) : (int64_t) sizeof(ggml_fp16_t);

    if (q->nb[0] != (int64_t) sizeof(float) || k->nb[0] != k_nb0 ||
        v->nb[0] != v_nb0 || dst->nb[0] != (int64_t) sizeof(float)) {
        debug_log("fallback: innermost stride is not contiguous");
        return false;
    }

    return true;
}

static void tensor_to_f32_contiguous(const ggml_tensor * src, std::vector<float> & dst) {
    const int64_t ne0 = src->ne[0];
    const int64_t ne1 = src->ne[1];
    const int64_t ne2 = src->ne[2];
    const int64_t ne3 = src->ne[3];

    dst.resize(ne0*ne1*ne2*ne3);

    const char * base = (const char *) src->data;
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const char * p = base + i0*src->nb[0] + i1*src->nb[1] + i2*src->nb[2] + i3*src->nb[3];
                    const int64_t j = i0 + ne0*(i1 + ne1*(i2 + ne2*i3));
                    if (src->type == GGML_TYPE_F32) {
                        dst[j] = *(const float *) p;
                    } else {
                        dst[j] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
                    }
                }
            }
        }
    }
}

static fused_cpp_sdpa_fn resolve() {
#if defined(_WIN32)
    return nullptr;
#else
    static bool tried = false;
    static fused_cpp_sdpa_fn fn = nullptr;

    if (tried) {
        return fn;
    }

    tried = true;

    const char * lib = std::getenv("GGML_FUSED_CPP_SDPA_LIB");
    if (lib == nullptr || lib[0] == '\0') {
        debug_log("fallback: GGML_FUSED_CPP_SDPA_LIB is not set");
        return nullptr;
    }

    void * handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        if (debug_enabled()) {
            std::fprintf(stderr, "GGML_FUSED_CPP_SDPA: dlopen failed: %s\n", dlerror());
        }
        return nullptr;
    }
    debug_log("dlopen ok");

    fn = reinterpret_cast<fused_cpp_sdpa_fn>(
            dlsym(handle, "fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp"));
    if (fn == nullptr && debug_enabled()) {
        std::fprintf(stderr, "GGML_FUSED_CPP_SDPA: dlsym failed: %s\n", dlerror());
    } else {
        debug_log("dlsym ok");
    }
    return fn;
#endif
}

} // namespace

bool ggml_fused_cpp_sdpa_flash_attn_ext_fp32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    if (!supported(params, dst)) {
        return false;
    }

    fused_cpp_sdpa_fn fn = resolve();
    if (fn == nullptr) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    std::vector<float> k_f32;
    std::vector<float> v_f32;

    const float * k_data = (const float *) k->data;
    const float * v_data = (const float *) v->data;

    int64_t k_nb0 = k->nb[0];
    int64_t k_nb1 = k->nb[1];
    int64_t k_nb2 = k->nb[2];
    int64_t k_nb3 = k->nb[3];
    int64_t v_nb0 = v->nb[0];
    int64_t v_nb1 = v->nb[1];
    int64_t v_nb2 = v->nb[2];
    int64_t v_nb3 = v->nb[3];

    if (k->type != GGML_TYPE_F32) {
        tensor_to_f32_contiguous(k, k_f32);
        k_data = k_f32.data();
        k_nb0 = (int64_t) sizeof(float);
        k_nb1 = k->ne[0]*k_nb0;
        k_nb2 = k->ne[1]*k_nb1;
        k_nb3 = k->ne[2]*k_nb2;
    }

    if (v->type != GGML_TYPE_F32) {
        tensor_to_f32_contiguous(v, v_f32);
        v_data = v_f32.data();
        v_nb0 = (int64_t) sizeof(float);
        v_nb1 = v->ne[0]*v_nb0;
        v_nb2 = v->ne[1]*v_nb1;
        v_nb3 = v->ne[2]*v_nb2;
    }

    const int rc = fn(
            (const float *) q->data,
            k_data,
            v_data,
            (float *) dst->data,
            q->ne[3],
            q->ne[2],
            q->ne[1],
            k->ne[1],
            q->ne[0],
            v->ne[0],
            q->nb[0], q->nb[1], q->nb[2], q->nb[3],
            k_nb0, k_nb1, k_nb2, k_nb3,
            v_nb0, v_nb1, v_nb2, v_nb3,
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
            get_f32_param(dst, 0));

    if (rc == 0) {
        debug_log("external sdpa used");
        return true;
    }

    if (debug_enabled()) {
        std::fprintf(stderr, "GGML_FUSED_CPP_SDPA: external sdpa returned rc=%d\n", rc);
    }
    return false;
}
