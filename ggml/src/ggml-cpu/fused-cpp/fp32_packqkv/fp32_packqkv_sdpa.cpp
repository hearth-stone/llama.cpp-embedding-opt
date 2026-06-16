#include "fp32_packqkv_sdpa.h"
#include "sdpa_microkernels/neon_cache_config.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

#if FUSED_CPP_SDPA_CACHE_HAS_SVE
#include <arm_sve.h>
#endif

#include "sdpa_common.h"
#include "sdpa_tile_sizes.h"
#include "sdpa_pack_utils.h"
#include "sdpa_flash2_neon_l3kv_impl.h"
#include "sdpa_microkernels/neon_cache_microkernels.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fp32_packqkv_sdpa {
namespace {

using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_taskloop;
using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes_l3kv;
using ::fused_cpp::sdpa_tile_sizes::effective_cache_bytes;
using ::fused_cpp::sdpa_tile_sizes::TileSizes;

constexpr int64_t kEvBlock = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct ElementStrides {
  int64_t b = 0;
  int64_t n = 0;
  int64_t t = 0;
  int64_t d = 1;
};

struct MaskF16Layout {
  int64_t ne0 = 0;
  int64_t ne1 = 0;
  int64_t ne2 = 0;
  int64_t ne3 = 0;
  int64_t nb0 = 0;
  int64_t nb1 = 0;
  int64_t nb2 = 0;
  int64_t nb3 = 0;
};

using MaskF32Layout = MaskF16Layout;

inline int64_t clamp_i64(int64_t v, int64_t lo, int64_t hi) {
  return std::max(lo, std::min(v, hi));
}

inline int64_t ceil_div8_i64(int64_t v) {
  return (v + 7) / 8;
}

bool env_flag_enabled(const char* name) {
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return false;
  }
  return std::strcmp(env, "0") != 0 &&
         std::strcmp(env, "off") != 0 &&
         std::strcmp(env, "false") != 0 &&
         std::strcmp(env, "no") != 0;
}

bool use_sve_microkernels() {
  static const bool enabled = env_flag_enabled("FUSED_CPP_SDPA_USE_SVE_KERNELS");
  return enabled;
}

// Runtime switch between the official qkt_8x8 (default) and the experimental
// test kernel. FUSED_CPP_SDPA_QKT_TEST=1 selects the test version for A/B
// perf comparison in a single binary.
bool use_qkt_8x8_test() {
  static const bool enabled = env_flag_enabled("FUSED_CPP_SDPA_QKT_TEST");
  return enabled;
}

inline void check_config(const Config& cfg) {
  if (cfg.B <= 0 || cfg.N <= 0 || cfg.L <= 0 || cfg.S <= 0 ||
      cfg.E <= 0 || cfg.Ev <= 0) {
    throw std::invalid_argument("shape dimensions must be positive");
  }
  if (cfg.Ev % kEvBlock != 0) {
    throw std::invalid_argument("fp32 packqkv path requires Ev % 8 == 0");
  }
  if (cfg.s_tile < 0 || cfg.s_tile % 8 != 0) {
    throw std::invalid_argument(
        "fp32 packqkv path requires s_tile == 0 or s_tile % 8 == 0");
  }
}

void pack_k_fp32_to_sblock8_one_head_strided(
    const float* k_src,
    float* k_dst,
    int64_t S,
    int64_t E,
    ElementStrides k_stride,
    float scale) {
  const int64_t S_blocks = ceil_div8_i64(S);
  const int64_t dst_stride_sb = E * 8;

  for (int64_t sb = 0; sb < S_blocks; ++sb) {
    const int64_t s0 = sb * 8;
    const int64_t remain = S - s0;
    const float* src = k_src + s0 * k_stride.t;
    float* dst = k_dst + sb * dst_stride_sb;
    for (int64_t e = 0; e < E; ++e) {
      float* d = dst + e * 8;
      const int64_t ed = e * k_stride.d;
      if (remain >= 8) {
        d[0] = src[0 * k_stride.t + ed] * scale;
        d[1] = src[1 * k_stride.t + ed] * scale;
        d[2] = src[2 * k_stride.t + ed] * scale;
        d[3] = src[3 * k_stride.t + ed] * scale;
        d[4] = src[4 * k_stride.t + ed] * scale;
        d[5] = src[5 * k_stride.t + ed] * scale;
        d[6] = src[6 * k_stride.t + ed] * scale;
        d[7] = src[7 * k_stride.t + ed] * scale;
      } else {
        const int64_t tail = std::max<int64_t>(0, remain);
        int64_t j = 0;
        for (; j + 4 <= tail; j += 4) {
          d[j + 0] = src[(j + 0) * k_stride.t + ed] * scale;
          d[j + 1] = src[(j + 1) * k_stride.t + ed] * scale;
          d[j + 2] = src[(j + 2) * k_stride.t + ed] * scale;
          d[j + 3] = src[(j + 3) * k_stride.t + ed] * scale;
        }
        for (; j + 2 <= tail; j += 2) {
          d[j + 0] = src[(j + 0) * k_stride.t + ed] * scale;
          d[j + 1] = src[(j + 1) * k_stride.t + ed] * scale;
        }
        for (; j < tail; ++j) {
          d[j] = src[j * k_stride.t + ed] * scale;
        }
        for (; j < 8; ++j) {
          d[j] = 0.0f;
        }
      }
    }
  }
}

void pack_k_fp32_to_sblock8_strided(
    const float* k_src,
    float* k_dst,
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t E,
    ElementStrides k_stride,
    float scale);

void pack_k_fp32_to_sblock8_strided(
    const float* k_src,
    float* k_dst,
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t E,
    ElementStrides k_stride,
    float scale) {
  const int64_t S_blocks = ceil_div8_i64(S);
  const int64_t dst_stride_b = N * S_blocks * E * 8;
  const int64_t dst_stride_n = S_blocks * E * 8;
  const int64_t dst_stride_sb = E * 8;

#ifdef _OPENMP
  #pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t sb = 0; sb < S_blocks; ++sb) {
        const int64_t s0 = sb * 8;
        const int64_t remain = S - s0;
        const float* src = k_src + b * k_stride.b
                                 + n * k_stride.n
                                 + s0 * k_stride.t;
        float* dst = k_dst + b * dst_stride_b
                           + n * dst_stride_n
                           + sb * dst_stride_sb;
        for (int64_t e = 0; e < E; ++e) {
          float* d = dst + e * 8;
          const int64_t ed = e * k_stride.d;
          if (remain >= 8) {
            d[0] = src[0 * k_stride.t + ed] * scale;
            d[1] = src[1 * k_stride.t + ed] * scale;
            d[2] = src[2 * k_stride.t + ed] * scale;
            d[3] = src[3 * k_stride.t + ed] * scale;
            d[4] = src[4 * k_stride.t + ed] * scale;
            d[5] = src[5 * k_stride.t + ed] * scale;
            d[6] = src[6 * k_stride.t + ed] * scale;
            d[7] = src[7 * k_stride.t + ed] * scale;
          } else {
            const int64_t tail = std::max<int64_t>(0, remain);
            int64_t j = 0;
            for (; j + 4 <= tail; j += 4) {
              d[j + 0] = src[(j + 0) * k_stride.t + ed] * scale;
              d[j + 1] = src[(j + 1) * k_stride.t + ed] * scale;
              d[j + 2] = src[(j + 2) * k_stride.t + ed] * scale;
              d[j + 3] = src[(j + 3) * k_stride.t + ed] * scale;
            }
            for (; j + 2 <= tail; j += 2) {
              d[j + 0] = src[(j + 0) * k_stride.t + ed] * scale;
              d[j + 1] = src[(j + 1) * k_stride.t + ed] * scale;
            }
            for (; j < tail; ++j) {
              d[j] = src[j * k_stride.t + ed] * scale;
            }
            for (; j < 8; ++j) {
              d[j] = 0.0f;
            }
          }
        }
      }
    }
  }
}

void pack_v_to_evblock8_one_head_strided(
    const float* v_src,
    float* v_dst,
    int64_t S,
    int64_t Ev,
    ElementStrides v_stride) {
  const int64_t Eb = Ev / 8;
  const int64_t evblock_stride_dst = S * 8;

  for (int64_t eb = 0; eb < Eb; ++eb) {
    const int64_t ev = eb * 8;
    const float* src = v_src + ev * v_stride.d;
    float* dst = v_dst + eb * evblock_stride_dst;
    for (int64_t s = 0; s < S; ++s) {
      const float* row = src + s * v_stride.t;
      float* packed = dst + s * 8;
      if (v_stride.d == 1) {
        ::fused_cpp::sdpa_pack_utils::copy_f32x8(row, packed);
      } else {
        for (int64_t i = 0; i < 8; ++i) {
          packed[i] = row[i * v_stride.d];
        }
      }
    }
  }
}

void pack_v_to_evblock8_strided(
    const float* v_src,
    float* v_dst,
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t Ev,
    ElementStrides v_stride) {
  const int64_t Eb = Ev / 8;
  const int64_t dst_stride_b = N * Eb * S * 8;
  const int64_t dst_stride_n = Eb * S * 8;
  const int64_t evblock_stride_dst = S * 8;

#ifdef _OPENMP
  #pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t eb = 0; eb < Eb; ++eb) {
        const int64_t ev = eb * 8;
        const float* src = v_src + b * v_stride.b
                                 + n * v_stride.n
                                 + ev * v_stride.d;
        float* dst = v_dst + b * dst_stride_b
                           + n * dst_stride_n
                           + eb * evblock_stride_dst;
        for (int64_t s = 0; s < S; ++s) {
          const float* row = src + s * v_stride.t;
          float* packed = dst + s * 8;
          if (v_stride.d == 1) {
            ::fused_cpp::sdpa_pack_utils::copy_f32x8(row, packed);
          } else {
            for (int64_t i = 0; i < 8; ++i) {
              packed[i] = row[i * v_stride.d];
            }
          }
        }
      }
    }
  }
}

void check_mask_layout(
    int64_t L,
    int64_t S,
    const MaskF16Layout& layout,
    const char* dtype_name) {
  if (layout.ne0 < S || layout.ne1 < L || layout.ne2 <= 0 || layout.ne3 <= 0) {
    throw std::invalid_argument(std::string("unsupported ") + dtype_name +
                                " mask shape");
  }
  if (layout.nb0 <= 0 || layout.nb1 <= 0 || layout.nb2 <= 0 || layout.nb3 <= 0) {
    throw std::invalid_argument(std::string("unsupported ") + dtype_name +
                                " mask strides");
  }
}

inline void qkt_packk8_tail_scalar(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float scale,
    float* scores_buf,
    int64_t scores_row_stride,
    int Lq,
    int Sk) {
  // scale is folded into packed K (see pack_k_fp32_to_sblock8_*), so the
  // dot product is already scaled here.
  (void)scale;
  for (int i = 0; i < Lq; ++i) {
    const float* q = Q + i * q_row_stride;
    float* out = scores_buf + i * scores_row_stride;
    for (int j = 0; j < Sk; ++j) {
      float sum = 0.0f;
      for (int64_t e = 0; e < E; ++e) {
        sum += q[e] * Kp[e * 8 + j];
      }
      out[j] = sum;
    }
  }
}

struct MK_Fp32PackK8PQuad {
  static constexpr const char* kName = "fp32_packk8_pquad";
  static constexpr bool kHasKSBlock8Layout = true;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  static constexpr bool kSupportsQktNeonRowMax = true;
#endif

  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    if (use_qkt_8x8_test()) {
      qkt_8x8_test(Q, q_row_stride, Kp, 0, E, scale, scores_buf, scores_row_stride);
      return;
    }
    float32x4_t lo0 = vdupq_n_f32(0.0f), hi0 = vdupq_n_f32(0.0f);
    float32x4_t lo1 = vdupq_n_f32(0.0f), hi1 = vdupq_n_f32(0.0f);
    float32x4_t lo2 = vdupq_n_f32(0.0f), hi2 = vdupq_n_f32(0.0f);
    float32x4_t lo3 = vdupq_n_f32(0.0f), hi3 = vdupq_n_f32(0.0f);
    float32x4_t lo4 = vdupq_n_f32(0.0f), hi4 = vdupq_n_f32(0.0f);
    float32x4_t lo5 = vdupq_n_f32(0.0f), hi5 = vdupq_n_f32(0.0f);
    float32x4_t lo6 = vdupq_n_f32(0.0f), hi6 = vdupq_n_f32(0.0f);
    float32x4_t lo7 = vdupq_n_f32(0.0f), hi7 = vdupq_n_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    // Keep the row-wise unroll-4 form: on Arm-codex, clang++-22 emits it
    // without vector-register moves and measured faster than GCC 13 for this
    // qkt_8x8 kernel. Source-level lane staging did not improve GCC output.
    int64_t e = 0;
    constexpr int64_t kQktPrefetchDistance = 16;
    for (; e + 4 <= E; e += 4) {
      const int64_t e_pf = e + kQktPrefetchDistance;
      if (e_pf < E) {
        const float* k_pf = Kp + e_pf * 8;
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf);
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf + 16);
      }

      const float32x4_t k0l = vld1q_f32(Kp + (e + 0) * 8 + 0);
      const float32x4_t k0h = vld1q_f32(Kp + (e + 0) * 8 + 4);
      const float32x4_t k1l = vld1q_f32(Kp + (e + 1) * 8 + 0);
      const float32x4_t k1h = vld1q_f32(Kp + (e + 1) * 8 + 4);
      const float32x4_t k2l = vld1q_f32(Kp + (e + 2) * 8 + 0);
      const float32x4_t k2h = vld1q_f32(Kp + (e + 2) * 8 + 4);
      const float32x4_t k3l = vld1q_f32(Kp + (e + 3) * 8 + 0);
      const float32x4_t k3h = vld1q_f32(Kp + (e + 3) * 8 + 4);
      const float32x4_t qv0 = vld1q_f32(q0 + e);
      const float32x4_t qv1 = vld1q_f32(q1 + e);
      const float32x4_t qv2 = vld1q_f32(q2 + e);
      const float32x4_t qv3 = vld1q_f32(q3 + e);
      const float32x4_t qv4 = vld1q_f32(q4 + e);
      const float32x4_t qv5 = vld1q_f32(q5 + e);
      const float32x4_t qv6 = vld1q_f32(q6 + e);
      const float32x4_t qv7 = vld1q_f32(q7 + e);

#define FUSED_CPP_QKT_PACKK8_FMA_ROW(ID, QV)                      \
      do {                                                        \
        lo##ID = vfmaq_laneq_f32(lo##ID, k0l, QV, 0);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k0h, QV, 0);             \
        lo##ID = vfmaq_laneq_f32(lo##ID, k1l, QV, 1);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k1h, QV, 1);             \
        lo##ID = vfmaq_laneq_f32(lo##ID, k2l, QV, 2);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k2h, QV, 2);             \
        lo##ID = vfmaq_laneq_f32(lo##ID, k3l, QV, 3);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k3h, QV, 3);             \
      } while (0)

      FUSED_CPP_QKT_PACKK8_FMA_ROW(0, qv0);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(1, qv1);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(2, qv2);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(3, qv3);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(4, qv4);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(5, qv5);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(6, qv6);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(7, qv7);
#undef FUSED_CPP_QKT_PACKK8_FMA_ROW
    }

    for (; e < E; ++e) {
      const float32x4_t kl = vld1q_f32(Kp + e * 8 + 0);
      const float32x4_t kh = vld1q_f32(Kp + e * 8 + 4);
#define FUSED_CPP_QKT_PACKK8_FMA_TAIL(ID, QROW)                    \
      do {                                                         \
        const float q = QROW[e];                                   \
        lo##ID = vfmaq_n_f32(lo##ID, kl, q);                       \
        hi##ID = vfmaq_n_f32(hi##ID, kh, q);                       \
      } while (0)
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(0, q0);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(1, q1);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(2, q2);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(3, q3);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(4, q4);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(5, q5);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(6, q6);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(7, q7);
#undef FUSED_CPP_QKT_PACKK8_FMA_TAIL
    }

    (void)scale;  // scale folded into packed K
#define FUSED_CPP_QKT_PACKK8_STORE_ROW(ID)                         \
    do {                                                           \
      vst1q_f32(scores_buf + (ID) * scores_row_stride + 0, lo##ID); \
      vst1q_f32(scores_buf + (ID) * scores_row_stride + 4, hi##ID); \
    } while (0)
    FUSED_CPP_QKT_PACKK8_STORE_ROW(0);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(1);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(2);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(3);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(4);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(5);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(6);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(7);
#undef FUSED_CPP_QKT_PACKK8_STORE_ROW
#else
    qkt_packk8_tail_scalar(
        Q, q_row_stride, Kp, E, scale, scores_buf, scores_row_stride, 8, 8);
#endif
  }

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  // Experimental qkt_8x8 (FUSED_CPP_SDPA_QKT_TEST=1). Three changes vs official:
  //  1. no zero-init: the first e-block seeds accumulators with vmul, then fmla.
  //  2. K side held in 4 registers (kAl/kAh/kBl/kBh), reloaded mid-block so K
  //     loads interleave with the FMAs instead of one big 8-register load.
  //  3. Q is single-buffered software-pipelined: the next block's Q is loaded
  //     one iteration ahead (16 accumulators leave no room to double-buffer all
  //     8 Q vectors, so a single rolling set is used).
  static inline void qkt_8x8_test(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    float32x4_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
    float32x4_t lo4, hi4, lo5, hi5, lo6, hi6, lo7, hi7;

    if (E >= 4) {
      float32x4_t kcur, knext;
      float32x4_t qv0 = vld1q_f32(q0 + 0);
      float32x4_t qv1 = vld1q_f32(q1 + 0);
      float32x4_t qv2 = vld1q_f32(q2 + 0);
      float32x4_t qv3 = vld1q_f32(q3 + 0);
      float32x4_t qv4 = vld1q_f32(q4 + 0);
      float32x4_t qv5 = vld1q_f32(q5 + 0);
      float32x4_t qv6 = vld1q_f32(q6 + 0);
      float32x4_t qv7 = vld1q_f32(q7 + 0);

      // K is a contiguous stream of 128-bit vectors (Kp + e*8 + N*4). Each
      // vector feeds 8 FMAs (one per row). It is held in a 2-register double
      // buffer: while the 8 FMAs run on kcur, the next vector is loaded into
      // knext, then swapped in. The 4-e block stays unrolled so the laneq
      // index is a compile-time constant.
#define FUSED_CPP_QKT_TEST_MUL(ACC, LANE)                     \
      do {                                                    \
        ACC##0 = vmulq_laneq_f32(kcur, qv0, LANE);            \
        ACC##1 = vmulq_laneq_f32(kcur, qv1, LANE);            \
        ACC##2 = vmulq_laneq_f32(kcur, qv2, LANE);            \
        ACC##3 = vmulq_laneq_f32(kcur, qv3, LANE);            \
        ACC##4 = vmulq_laneq_f32(kcur, qv4, LANE);            \
        ACC##5 = vmulq_laneq_f32(kcur, qv5, LANE);            \
        ACC##6 = vmulq_laneq_f32(kcur, qv6, LANE);            \
        ACC##7 = vmulq_laneq_f32(kcur, qv7, LANE);            \
      } while (0)
#define FUSED_CPP_QKT_TEST_FMA(ACC, LANE)                     \
      do {                                                    \
        ACC##0 = vfmaq_laneq_f32(ACC##0, kcur, qv0, LANE);    \
        ACC##1 = vfmaq_laneq_f32(ACC##1, kcur, qv1, LANE);    \
        ACC##2 = vfmaq_laneq_f32(ACC##2, kcur, qv2, LANE);    \
        ACC##3 = vfmaq_laneq_f32(ACC##3, kcur, qv3, LANE);    \
        ACC##4 = vfmaq_laneq_f32(ACC##4, kcur, qv4, LANE);    \
        ACC##5 = vfmaq_laneq_f32(ACC##5, kcur, qv5, LANE);    \
        ACC##6 = vfmaq_laneq_f32(ACC##6, kcur, qv6, LANE);    \
        ACC##7 = vfmaq_laneq_f32(ACC##7, kcur, qv7, LANE);    \
      } while (0)

      // ---- first block e=0..3: seed lo/hi (lane 0) with vmul, rest fmla ----
      kcur  = vld1q_f32(Kp + 0);
      knext = vld1q_f32(Kp + 4);   FUSED_CPP_QKT_TEST_MUL(lo, 0); kcur = knext;
      knext = vld1q_f32(Kp + 8);   FUSED_CPP_QKT_TEST_MUL(hi, 0); kcur = knext;
      knext = vld1q_f32(Kp + 12);  FUSED_CPP_QKT_TEST_FMA(lo, 1); kcur = knext;
      knext = vld1q_f32(Kp + 16);  FUSED_CPP_QKT_TEST_FMA(hi, 1); kcur = knext;
      knext = vld1q_f32(Kp + 20);  FUSED_CPP_QKT_TEST_FMA(lo, 2); kcur = knext;
      knext = vld1q_f32(Kp + 24);  FUSED_CPP_QKT_TEST_FMA(hi, 2); kcur = knext;
      knext = vld1q_f32(Kp + 28);  FUSED_CPP_QKT_TEST_FMA(lo, 3); kcur = knext;
      FUSED_CPP_QKT_TEST_FMA(hi, 3);

      int64_t e = 4;
      // single-buffered Q software pipeline: preload first main block's Q
      if (e + 4 <= E) {
        qv0 = vld1q_f32(q0 + e); qv1 = vld1q_f32(q1 + e);
        qv2 = vld1q_f32(q2 + e); qv3 = vld1q_f32(q3 + e);
        qv4 = vld1q_f32(q4 + e); qv5 = vld1q_f32(q5 + e);
        qv6 = vld1q_f32(q6 + e); qv7 = vld1q_f32(q7 + e);
      }

      constexpr int64_t kQktPrefetchDistance = 16;
      for (; e + 4 <= E; e += 4) {
        const int64_t e_pf = e + kQktPrefetchDistance;
        if (e_pf < E) {
          const float* k_pf = Kp + e_pf * 8;
          ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf);
          ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf + 16);
        }

        const float* kb = Kp + e * 8;
        kcur  = vld1q_f32(kb + 0);
        knext = vld1q_f32(kb + 4);   FUSED_CPP_QKT_TEST_FMA(lo, 0); kcur = knext;
        knext = vld1q_f32(kb + 8);   FUSED_CPP_QKT_TEST_FMA(hi, 0); kcur = knext;
        knext = vld1q_f32(kb + 12);  FUSED_CPP_QKT_TEST_FMA(lo, 1); kcur = knext;
        knext = vld1q_f32(kb + 16);  FUSED_CPP_QKT_TEST_FMA(hi, 1); kcur = knext;
        knext = vld1q_f32(kb + 20);  FUSED_CPP_QKT_TEST_FMA(lo, 2); kcur = knext;
        knext = vld1q_f32(kb + 24);  FUSED_CPP_QKT_TEST_FMA(hi, 2); kcur = knext;
        knext = vld1q_f32(kb + 28);  FUSED_CPP_QKT_TEST_FMA(lo, 3); kcur = knext;
        FUSED_CPP_QKT_TEST_FMA(hi, 3);

        // Q pipeline: preload next block's Q one iteration ahead
        if (e + 8 <= E) {
          qv0 = vld1q_f32(q0 + e + 4); qv1 = vld1q_f32(q1 + e + 4);
          qv2 = vld1q_f32(q2 + e + 4); qv3 = vld1q_f32(q3 + e + 4);
          qv4 = vld1q_f32(q4 + e + 4); qv5 = vld1q_f32(q5 + e + 4);
          qv6 = vld1q_f32(q6 + e + 4); qv7 = vld1q_f32(q7 + e + 4);
        }
      }
#undef FUSED_CPP_QKT_TEST_MUL
#undef FUSED_CPP_QKT_TEST_FMA

      // scalar tail for E % 4 != 0
      for (; e < E; ++e) {
        const float32x4_t kl = vld1q_f32(Kp + e * 8 + 0);
        const float32x4_t kh = vld1q_f32(Kp + e * 8 + 4);
#define FUSED_CPP_QKT_TEST_TAIL(ID, QROW)                     \
        do {                                                  \
          const float qf = QROW[e];                           \
          lo##ID = vfmaq_n_f32(lo##ID, kl, qf);               \
          hi##ID = vfmaq_n_f32(hi##ID, kh, qf);               \
        } while (0)
        FUSED_CPP_QKT_TEST_TAIL(0, q0);
        FUSED_CPP_QKT_TEST_TAIL(1, q1);
        FUSED_CPP_QKT_TEST_TAIL(2, q2);
        FUSED_CPP_QKT_TEST_TAIL(3, q3);
        FUSED_CPP_QKT_TEST_TAIL(4, q4);
        FUSED_CPP_QKT_TEST_TAIL(5, q5);
        FUSED_CPP_QKT_TEST_TAIL(6, q6);
        FUSED_CPP_QKT_TEST_TAIL(7, q7);
#undef FUSED_CPP_QKT_TEST_TAIL
      }
    } else {
      // rare path E < 4: zero-init + scalar accumulate
      lo0 = vdupq_n_f32(0.0f); hi0 = vdupq_n_f32(0.0f);
      lo1 = vdupq_n_f32(0.0f); hi1 = vdupq_n_f32(0.0f);
      lo2 = vdupq_n_f32(0.0f); hi2 = vdupq_n_f32(0.0f);
      lo3 = vdupq_n_f32(0.0f); hi3 = vdupq_n_f32(0.0f);
      lo4 = vdupq_n_f32(0.0f); hi4 = vdupq_n_f32(0.0f);
      lo5 = vdupq_n_f32(0.0f); hi5 = vdupq_n_f32(0.0f);
      lo6 = vdupq_n_f32(0.0f); hi6 = vdupq_n_f32(0.0f);
      lo7 = vdupq_n_f32(0.0f); hi7 = vdupq_n_f32(0.0f);
      for (int64_t e = 0; e < E; ++e) {
        const float32x4_t kl = vld1q_f32(Kp + e * 8 + 0);
        const float32x4_t kh = vld1q_f32(Kp + e * 8 + 4);
#define FUSED_CPP_QKT_TEST_SMALL(ID, QROW)                    \
        do {                                                  \
          const float qf = QROW[e];                           \
          lo##ID = vfmaq_n_f32(lo##ID, kl, qf);               \
          hi##ID = vfmaq_n_f32(hi##ID, kh, qf);               \
        } while (0)
        FUSED_CPP_QKT_TEST_SMALL(0, q0);
        FUSED_CPP_QKT_TEST_SMALL(1, q1);
        FUSED_CPP_QKT_TEST_SMALL(2, q2);
        FUSED_CPP_QKT_TEST_SMALL(3, q3);
        FUSED_CPP_QKT_TEST_SMALL(4, q4);
        FUSED_CPP_QKT_TEST_SMALL(5, q5);
        FUSED_CPP_QKT_TEST_SMALL(6, q6);
        FUSED_CPP_QKT_TEST_SMALL(7, q7);
#undef FUSED_CPP_QKT_TEST_SMALL
      }
    }

    (void)scale;  // scale folded into packed K
#define FUSED_CPP_QKT_TEST_STORE(ID)                                   \
    do {                                                               \
      vst1q_f32(scores_buf + (ID) * scores_row_stride + 0, lo##ID);    \
      vst1q_f32(scores_buf + (ID) * scores_row_stride + 4, hi##ID);    \
    } while (0)
    FUSED_CPP_QKT_TEST_STORE(0);
    FUSED_CPP_QKT_TEST_STORE(1);
    FUSED_CPP_QKT_TEST_STORE(2);
    FUSED_CPP_QKT_TEST_STORE(3);
    FUSED_CPP_QKT_TEST_STORE(4);
    FUSED_CPP_QKT_TEST_STORE(5);
    FUSED_CPP_QKT_TEST_STORE(6);
    FUSED_CPP_QKT_TEST_STORE(7);
#undef FUSED_CPP_QKT_TEST_STORE
  }

  static inline void qkt_8x8_rowmax(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      float32x4_t row_max_lo[8],
      float32x4_t row_max_hi[8]) {
    float32x4_t lo0 = vdupq_n_f32(0.0f), hi0 = vdupq_n_f32(0.0f);
    float32x4_t lo1 = vdupq_n_f32(0.0f), hi1 = vdupq_n_f32(0.0f);
    float32x4_t lo2 = vdupq_n_f32(0.0f), hi2 = vdupq_n_f32(0.0f);
    float32x4_t lo3 = vdupq_n_f32(0.0f), hi3 = vdupq_n_f32(0.0f);
    float32x4_t lo4 = vdupq_n_f32(0.0f), hi4 = vdupq_n_f32(0.0f);
    float32x4_t lo5 = vdupq_n_f32(0.0f), hi5 = vdupq_n_f32(0.0f);
    float32x4_t lo6 = vdupq_n_f32(0.0f), hi6 = vdupq_n_f32(0.0f);
    float32x4_t lo7 = vdupq_n_f32(0.0f), hi7 = vdupq_n_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    int64_t e = 0;
    constexpr int64_t kQktPrefetchDistance = 16;
    for (; e + 4 <= E; e += 4) {
      const int64_t e_pf = e + kQktPrefetchDistance;
      if (e_pf < E) {
        const float* k_pf = Kp + e_pf * 8;
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf);
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf + 16);
      }

      const float32x4_t k0l = vld1q_f32(Kp + (e + 0) * 8 + 0);
      const float32x4_t k0h = vld1q_f32(Kp + (e + 0) * 8 + 4);
      const float32x4_t k1l = vld1q_f32(Kp + (e + 1) * 8 + 0);
      const float32x4_t k1h = vld1q_f32(Kp + (e + 1) * 8 + 4);
      const float32x4_t k2l = vld1q_f32(Kp + (e + 2) * 8 + 0);
      const float32x4_t k2h = vld1q_f32(Kp + (e + 2) * 8 + 4);
      const float32x4_t k3l = vld1q_f32(Kp + (e + 3) * 8 + 0);
      const float32x4_t k3h = vld1q_f32(Kp + (e + 3) * 8 + 4);
      const float32x4_t qv0 = vld1q_f32(q0 + e);
      const float32x4_t qv1 = vld1q_f32(q1 + e);
      const float32x4_t qv2 = vld1q_f32(q2 + e);
      const float32x4_t qv3 = vld1q_f32(q3 + e);
      const float32x4_t qv4 = vld1q_f32(q4 + e);
      const float32x4_t qv5 = vld1q_f32(q5 + e);
      const float32x4_t qv6 = vld1q_f32(q6 + e);
      const float32x4_t qv7 = vld1q_f32(q7 + e);

#define FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(ID, QV)               \
      do {                                                        \
        lo##ID = vfmaq_laneq_f32(lo##ID, k0l, QV, 0);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k0h, QV, 0);             \
        lo##ID = vfmaq_laneq_f32(lo##ID, k1l, QV, 1);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k1h, QV, 1);             \
        lo##ID = vfmaq_laneq_f32(lo##ID, k2l, QV, 2);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k2h, QV, 2);             \
        lo##ID = vfmaq_laneq_f32(lo##ID, k3l, QV, 3);             \
        hi##ID = vfmaq_laneq_f32(hi##ID, k3h, QV, 3);             \
      } while (0)

      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(0, qv0);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(1, qv1);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(2, qv2);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(3, qv3);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(4, qv4);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(5, qv5);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(6, qv6);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW(7, qv7);
#undef FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_ROW
    }

    for (; e < E; ++e) {
      const float32x4_t kl = vld1q_f32(Kp + e * 8 + 0);
      const float32x4_t kh = vld1q_f32(Kp + e * 8 + 4);
#define FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(ID, QROW)            \
      do {                                                        \
        const float q = QROW[e];                                  \
        lo##ID = vfmaq_n_f32(lo##ID, kl, q);                      \
        hi##ID = vfmaq_n_f32(hi##ID, kh, q);                      \
      } while (0)
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(0, q0);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(1, q1);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(2, q2);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(3, q3);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(4, q4);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(5, q5);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(6, q6);
      FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL(7, q7);
#undef FUSED_CPP_QKT_PACKK8_ROWMAX_FMA_TAIL
    }

    (void)scale;  // scale folded into packed K
#define FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(ID)                 \
    do {                                                          \
      row_max_lo[ID] = vmaxq_f32(row_max_lo[ID], lo##ID);         \
      row_max_hi[ID] = vmaxq_f32(row_max_hi[ID], hi##ID);         \
      vst1q_f32(scores_buf + (ID) * scores_row_stride + 0, lo##ID); \
      vst1q_f32(scores_buf + (ID) * scores_row_stride + 4, hi##ID); \
    } while (0)
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(0);
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(1);
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(2);
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(3);
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(4);
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(5);
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(6);
    FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW(7);
#undef FUSED_CPP_QKT_PACKK8_ROWMAX_STORE_ROW
  }
#endif

  static inline void qkt_8x4(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    float32x4_t acc4 = vdupq_n_f32(0.0f);
    float32x4_t acc5 = vdupq_n_f32(0.0f);
    float32x4_t acc6 = vdupq_n_f32(0.0f);
    float32x4_t acc7 = vdupq_n_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    int64_t e = 0;
    for (; e + 4 <= E; e += 4) {
      const float32x4_t qv0 = vld1q_f32(q0 + e);
      const float32x4_t qv1 = vld1q_f32(q1 + e);
      const float32x4_t qv2 = vld1q_f32(q2 + e);
      const float32x4_t qv3 = vld1q_f32(q3 + e);
      const float32x4_t qv4 = vld1q_f32(q4 + e);
      const float32x4_t qv5 = vld1q_f32(q5 + e);
      const float32x4_t qv6 = vld1q_f32(q6 + e);
      const float32x4_t qv7 = vld1q_f32(q7 + e);
      const float32x4_t k0 = vld1q_f32(Kp + (e + 0) * 8);
      const float32x4_t k1 = vld1q_f32(Kp + (e + 1) * 8);
      const float32x4_t k2 = vld1q_f32(Kp + (e + 2) * 8);
      const float32x4_t k3 = vld1q_f32(Kp + (e + 3) * 8);
#define FUSED_CPP_QKT_PACKK4_FMA_ROW(ID, QV)                       \
      do {                                                         \
        acc##ID = vfmaq_laneq_f32(acc##ID, k0, QV, 0);             \
        acc##ID = vfmaq_laneq_f32(acc##ID, k1, QV, 1);             \
        acc##ID = vfmaq_laneq_f32(acc##ID, k2, QV, 2);             \
        acc##ID = vfmaq_laneq_f32(acc##ID, k3, QV, 3);             \
      } while (0)
      FUSED_CPP_QKT_PACKK4_FMA_ROW(0, qv0);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(1, qv1);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(2, qv2);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(3, qv3);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(4, qv4);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(5, qv5);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(6, qv6);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(7, qv7);
#undef FUSED_CPP_QKT_PACKK4_FMA_ROW
    }
    for (; e < E; ++e) {
      const float32x4_t k = vld1q_f32(Kp + e * 8);
#define FUSED_CPP_QKT_PACKK4_FMA_TAIL(ID, QROW)                    \
      do { acc##ID = vfmaq_n_f32(acc##ID, k, QROW[e]); } while (0)
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(0, q0);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(1, q1);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(2, q2);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(3, q3);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(4, q4);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(5, q5);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(6, q6);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(7, q7);
#undef FUSED_CPP_QKT_PACKK4_FMA_TAIL
    }
    (void)scale;  // scale folded into packed K
#define FUSED_CPP_QKT_PACKK4_STORE_ROW(ID)                         \
    vst1q_f32(scores_buf + (ID) * scores_row_stride, acc##ID)
    FUSED_CPP_QKT_PACKK4_STORE_ROW(0);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(1);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(2);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(3);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(4);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(5);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(6);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(7);
#undef FUSED_CPP_QKT_PACKK4_STORE_ROW
#else
    qkt_packk8_tail_scalar(
        Q, q_row_stride, Kp, E, scale, scores_buf, scores_row_stride, 8, 4);
#endif
  }

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  static inline void qkt_8x4_rowmax(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      float32x4_t row_max_lo[8]) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    float32x4_t acc4 = vdupq_n_f32(0.0f);
    float32x4_t acc5 = vdupq_n_f32(0.0f);
    float32x4_t acc6 = vdupq_n_f32(0.0f);
    float32x4_t acc7 = vdupq_n_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    int64_t e = 0;
    for (; e + 4 <= E; e += 4) {
      const float32x4_t qv0 = vld1q_f32(q0 + e);
      const float32x4_t qv1 = vld1q_f32(q1 + e);
      const float32x4_t qv2 = vld1q_f32(q2 + e);
      const float32x4_t qv3 = vld1q_f32(q3 + e);
      const float32x4_t qv4 = vld1q_f32(q4 + e);
      const float32x4_t qv5 = vld1q_f32(q5 + e);
      const float32x4_t qv6 = vld1q_f32(q6 + e);
      const float32x4_t qv7 = vld1q_f32(q7 + e);
      const float32x4_t k0 = vld1q_f32(Kp + (e + 0) * 8);
      const float32x4_t k1 = vld1q_f32(Kp + (e + 1) * 8);
      const float32x4_t k2 = vld1q_f32(Kp + (e + 2) * 8);
      const float32x4_t k3 = vld1q_f32(Kp + (e + 3) * 8);
#define FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(ID, QV)               \
      do {                                                        \
        acc##ID = vfmaq_laneq_f32(acc##ID, k0, QV, 0);            \
        acc##ID = vfmaq_laneq_f32(acc##ID, k1, QV, 1);            \
        acc##ID = vfmaq_laneq_f32(acc##ID, k2, QV, 2);            \
        acc##ID = vfmaq_laneq_f32(acc##ID, k3, QV, 3);            \
      } while (0)
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(0, qv0);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(1, qv1);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(2, qv2);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(3, qv3);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(4, qv4);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(5, qv5);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(6, qv6);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW(7, qv7);
#undef FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_ROW
    }
    for (; e < E; ++e) {
      const float32x4_t k = vld1q_f32(Kp + e * 8);
#define FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(ID, QROW)            \
      do { acc##ID = vfmaq_n_f32(acc##ID, k, QROW[e]); } while (0)
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(0, q0);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(1, q1);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(2, q2);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(3, q3);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(4, q4);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(5, q5);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(6, q6);
      FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL(7, q7);
#undef FUSED_CPP_QKT_PACKK4_ROWMAX_FMA_TAIL
    }

    (void)scale;  // scale folded into packed K
#define FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(ID)                 \
    do {                                                          \
      row_max_lo[ID] = vmaxq_f32(row_max_lo[ID], acc##ID);        \
      vst1q_f32(scores_buf + (ID) * scores_row_stride, acc##ID);  \
    } while (0)
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(0);
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(1);
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(2);
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(3);
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(4);
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(5);
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(6);
    FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW(7);
#undef FUSED_CPP_QKT_PACKK4_ROWMAX_STORE_ROW
  }
#endif

  static inline void qkt_tail(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    qkt_packk8_tail_scalar(
        Q, q_row_stride, Kp, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }

  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    ::fused_cpp::sdpa_microkernels::gemm_pv_microkernel_8x8_fp32_pquad(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    ::fused_cpp::sdpa_microkernels::gemm_pv_8x8(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }

  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    ::fused_cpp::sdpa_microkernels::gemm_pv_tail(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
};

#if FUSED_CPP_SDPA_CACHE_HAS_SVE
struct MK_Fp32PackK8PQuadSve : MK_Fp32PackK8PQuad {
  static constexpr const char* kName = "fp32_packk8_pquad_sve";
  static constexpr bool kSupportsQktNeonRowMax = false;

  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    if (svcntw() < 8) {
      MK_Fp32PackK8PQuad::qkt_8x8(
          Q, q_row_stride, Kp, 0, E, scale, scores_buf, scores_row_stride);
      return;
    }

    const svbool_t pg = svptrue_pat_b32(SV_VL8);
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);
    svfloat32_t acc4 = svdup_f32(0.0f);
    svfloat32_t acc5 = svdup_f32(0.0f);
    svfloat32_t acc6 = svdup_f32(0.0f);
    svfloat32_t acc7 = svdup_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    int64_t e = 0;
    constexpr int64_t kQktPrefetchDistance = 16;
#define FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(ID, QV)                  \
      do {                                                        \
        acc##ID = svmla_lane_f32(acc##ID, k0, QV, 0);             \
        acc##ID = svmla_lane_f32(acc##ID, k1, QV, 1);             \
        acc##ID = svmla_lane_f32(acc##ID, k2, QV, 2);             \
        acc##ID = svmla_lane_f32(acc##ID, k3, QV, 3);             \
      } while (0)

    for (; e + 4 <= E; e += 4) {
      const int64_t e_pf = e + kQktPrefetchDistance;
      if (e_pf < E) {
        const float* k_pf = Kp + e_pf * 8;
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf);
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(k_pf + 16);
      }

      const svfloat32_t k0 = svld1_f32(pg, Kp + (e + 0) * 8);
      const svfloat32_t k1 = svld1_f32(pg, Kp + (e + 1) * 8);
      const svfloat32_t k2 = svld1_f32(pg, Kp + (e + 2) * 8);
      const svfloat32_t k3 = svld1_f32(pg, Kp + (e + 3) * 8);
      const svfloat32_t qv0 = svld1rq_f32(pg, q0 + e);
      const svfloat32_t qv1 = svld1rq_f32(pg, q1 + e);
      const svfloat32_t qv2 = svld1rq_f32(pg, q2 + e);
      const svfloat32_t qv3 = svld1rq_f32(pg, q3 + e);
      const svfloat32_t qv4 = svld1rq_f32(pg, q4 + e);
      const svfloat32_t qv5 = svld1rq_f32(pg, q5 + e);
      const svfloat32_t qv6 = svld1rq_f32(pg, q6 + e);
      const svfloat32_t qv7 = svld1rq_f32(pg, q7 + e);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(0, qv0);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(1, qv1);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(2, qv2);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(3, qv3);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(4, qv4);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(5, qv5);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(6, qv6);
      FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW(7, qv7);
    }
#undef FUSED_CPP_QKT_PACKK8_SVE_FMA_ROW

    for (; e < E; ++e) {
      const svfloat32_t k = svld1_f32(pg, Kp + e * 8);
#define FUSED_CPP_QKT_PACKK8_SVE_TAIL(ID, QROW)                    \
      acc##ID = svmla_n_f32_x(pg, acc##ID, k, QROW[e])
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(0, q0);
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(1, q1);
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(2, q2);
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(3, q3);
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(4, q4);
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(5, q5);
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(6, q6);
      FUSED_CPP_QKT_PACKK8_SVE_TAIL(7, q7);
#undef FUSED_CPP_QKT_PACKK8_SVE_TAIL
    }

    // scale folded into packed K
#define FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(ID)                     \
    svst1_f32(pg, scores_buf + (ID) * scores_row_stride, acc##ID)
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(0);
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(1);
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(2);
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(3);
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(4);
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(5);
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(6);
    FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW(7);
#undef FUSED_CPP_QKT_PACKK8_SVE_STORE_ROW
  }

  static inline void qkt_4x32(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    if (svcntw() < 8) {
      for (int block = 0; block < 4; ++block) {
        qkt_packk8_tail_scalar(
            Q, q_row_stride, Kp + block * E * 8, E, scale,
            scores_buf + block * 8, scores_row_stride, 4, 8);
      }
      return;
    }

    const svbool_t pg = svptrue_pat_b32(SV_VL8);
    svfloat32_t c00 = svdup_f32(0.0f);
    svfloat32_t c01 = svdup_f32(0.0f);
    svfloat32_t c02 = svdup_f32(0.0f);
    svfloat32_t c03 = svdup_f32(0.0f);
    svfloat32_t c10 = svdup_f32(0.0f);
    svfloat32_t c11 = svdup_f32(0.0f);
    svfloat32_t c12 = svdup_f32(0.0f);
    svfloat32_t c13 = svdup_f32(0.0f);
    svfloat32_t c20 = svdup_f32(0.0f);
    svfloat32_t c21 = svdup_f32(0.0f);
    svfloat32_t c22 = svdup_f32(0.0f);
    svfloat32_t c23 = svdup_f32(0.0f);
    svfloat32_t c30 = svdup_f32(0.0f);
    svfloat32_t c31 = svdup_f32(0.0f);
    svfloat32_t c32 = svdup_f32(0.0f);
    svfloat32_t c33 = svdup_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* k0_base = Kp + 0 * E * 8;
    const float* k1_base = Kp + 1 * E * 8;
    const float* k2_base = Kp + 2 * E * 8;
    const float* k3_base = Kp + 3 * E * 8;

#define FUSED_CPP_QKT_4X32_SVE_APPLY(ID, A, B0, B1, B2, B3)       \
    do {                                                          \
      c##ID##0 = svmla_f32_x(pg, c##ID##0, B0, A);                \
      c##ID##1 = svmla_f32_x(pg, c##ID##1, B1, A);                \
      c##ID##2 = svmla_f32_x(pg, c##ID##2, B2, A);                \
      c##ID##3 = svmla_f32_x(pg, c##ID##3, B3, A);                \
    } while (0)

#define FUSED_CPP_QKT_4X32_SVE_PAIR(EA, EB)                       \
    do {                                                          \
      const int64_t ea = (EA);                                    \
      const int64_t eb = (EB);                                    \
      const svfloat32_t b0 = svld1_f32(pg, k0_base + ea * 8);     \
      const svfloat32_t b1 = svld1_f32(pg, k1_base + ea * 8);     \
      const svfloat32_t b2 = svld1_f32(pg, k2_base + ea * 8);     \
      const svfloat32_t b3 = svld1_f32(pg, k3_base + ea * 8);     \
      const svfloat32_t n0 = svld1_f32(pg, k0_base + eb * 8);     \
      const svfloat32_t n1 = svld1_f32(pg, k1_base + eb * 8);     \
      const svfloat32_t n2 = svld1_f32(pg, k2_base + eb * 8);     \
      const svfloat32_t n3 = svld1_f32(pg, k3_base + eb * 8);     \
      const svfloat32_t a0 = svdup_f32(q0[ea]);                   \
      const svfloat32_t a1 = svdup_f32(q1[ea]);                   \
      const svfloat32_t a2 = svdup_f32(q2[ea]);                   \
      const svfloat32_t a3 = svdup_f32(q3[ea]);                   \
      FUSED_CPP_QKT_4X32_SVE_APPLY(0, a0, b0, b1, b2, b3);        \
      FUSED_CPP_QKT_4X32_SVE_APPLY(1, a1, b0, b1, b2, b3);        \
      FUSED_CPP_QKT_4X32_SVE_APPLY(2, a2, b0, b1, b2, b3);        \
      FUSED_CPP_QKT_4X32_SVE_APPLY(3, a3, b0, b1, b2, b3);        \
      const svfloat32_t an0 = svdup_f32(q0[eb]);                  \
      const svfloat32_t an1 = svdup_f32(q1[eb]);                  \
      const svfloat32_t an2 = svdup_f32(q2[eb]);                  \
      const svfloat32_t an3 = svdup_f32(q3[eb]);                  \
      FUSED_CPP_QKT_4X32_SVE_APPLY(0, an0, n0, n1, n2, n3);       \
      FUSED_CPP_QKT_4X32_SVE_APPLY(1, an1, n0, n1, n2, n3);       \
      FUSED_CPP_QKT_4X32_SVE_APPLY(2, an2, n0, n1, n2, n3);       \
      FUSED_CPP_QKT_4X32_SVE_APPLY(3, an3, n0, n1, n2, n3);       \
    } while (0)

#define FUSED_CPP_QKT_4X32_SVE_STEP(EE)                           \
    do {                                                          \
      const int64_t ee = (EE);                                    \
      const svfloat32_t b0 = svld1_f32(pg, k0_base + ee * 8);     \
      const svfloat32_t b1 = svld1_f32(pg, k1_base + ee * 8);     \
      const svfloat32_t b2 = svld1_f32(pg, k2_base + ee * 8);     \
      const svfloat32_t b3 = svld1_f32(pg, k3_base + ee * 8);     \
      const svfloat32_t a0 = svdup_f32(q0[ee]);                   \
      const svfloat32_t a1 = svdup_f32(q1[ee]);                   \
      const svfloat32_t a2 = svdup_f32(q2[ee]);                   \
      const svfloat32_t a3 = svdup_f32(q3[ee]);                   \
      FUSED_CPP_QKT_4X32_SVE_APPLY(0, a0, b0, b1, b2, b3);        \
      FUSED_CPP_QKT_4X32_SVE_APPLY(1, a1, b0, b1, b2, b3);        \
      FUSED_CPP_QKT_4X32_SVE_APPLY(2, a2, b0, b1, b2, b3);        \
      FUSED_CPP_QKT_4X32_SVE_APPLY(3, a3, b0, b1, b2, b3);        \
    } while (0)

    int64_t e = 0;
    constexpr int64_t kQktPrefetchDistance = 16;
    for (; e + 8 <= E; e += 8) {
      const int64_t e_pf = e + kQktPrefetchDistance;
      if (e_pf < E) {
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(
            k0_base + e_pf * 8);
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(
            k1_base + e_pf * 8);
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(
            k2_base + e_pf * 8);
        ::fused_cpp::sdpa_flash2_neon_l3kv_impl::prefetch_l1_keep_impl(
            k3_base + e_pf * 8);
      }
      FUSED_CPP_QKT_4X32_SVE_PAIR(e + 0, e + 1);
      FUSED_CPP_QKT_4X32_SVE_PAIR(e + 2, e + 3);
      FUSED_CPP_QKT_4X32_SVE_PAIR(e + 4, e + 5);
      FUSED_CPP_QKT_4X32_SVE_PAIR(e + 6, e + 7);
    }
    for (; e < E; ++e) {
      FUSED_CPP_QKT_4X32_SVE_STEP(e);
    }

    // scale folded into packed K
#define FUSED_CPP_QKT_4X32_SVE_STORE_ROW(ID)                      \
    do {                                                          \
      svst1_f32(pg, scores_buf + (ID) * scores_row_stride + 0,  c##ID##0); \
      svst1_f32(pg, scores_buf + (ID) * scores_row_stride + 8,  c##ID##1); \
      svst1_f32(pg, scores_buf + (ID) * scores_row_stride + 16, c##ID##2); \
      svst1_f32(pg, scores_buf + (ID) * scores_row_stride + 24, c##ID##3); \
    } while (0)
    FUSED_CPP_QKT_4X32_SVE_STORE_ROW(0);
    FUSED_CPP_QKT_4X32_SVE_STORE_ROW(1);
    FUSED_CPP_QKT_4X32_SVE_STORE_ROW(2);
    FUSED_CPP_QKT_4X32_SVE_STORE_ROW(3);
#undef FUSED_CPP_QKT_4X32_SVE_STORE_ROW
#undef FUSED_CPP_QKT_4X32_SVE_STEP
#undef FUSED_CPP_QKT_4X32_SVE_PAIR
#undef FUSED_CPP_QKT_4X32_SVE_APPLY
  }

  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    if (!::fused_cpp::sdpa_microkernels::gemm_pv_microkernel_8x8_fp32_pquad_sve(
            P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride)) {
      MK_Fp32PackK8PQuad::pv_8x8(
          P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
    }
  }

  static inline void pv_8x16(
      const float* P_hat, int64_t P_row_stride,
      const float* V_lo, const float* V_hi,
      int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    if (!::fused_cpp::sdpa_microkernels::gemm_pv_microkernel_8x16_fp32_pquad_sve(
            P_hat, P_row_stride, V_lo, V_hi, v_row_stride, Sk, O, o_row_stride)) {
      MK_Fp32PackK8PQuad::pv_8x8(
          P_hat, P_row_stride, V_lo, v_row_stride, Sk, O, o_row_stride);
      MK_Fp32PackK8PQuad::pv_8x8(
          P_hat, P_row_stride, V_hi, v_row_stride, Sk, O + 8, o_row_stride);
    }
  }
};
#endif

template <typename MK, bool kCausal, bool kHasMask>
void run_fp32_packk_path(
    const float* q_ptr,
    const float* k_packed_ptr,
    const float* v_packed_ptr,
    const SdpaParams& p,
    const TileSizes& ts,
    bool path_a,
    int total_threads,
    ElementStrides q_stride,
    ElementStrides out_stride) {
  const int64_t S_blocks = ceil_div8_i64(p.S);
  const int64_t k_packed_stride_n = S_blocks * p.E * 8;
  const int64_t k_packed_stride_b = p.N * k_packed_stride_n;
  const int64_t k_packed_stride_s = p.E;

  const int64_t Eb = p.Ev / 8;
  const int64_t v_evblock_stride = p.S * 8;
  const int64_t v_packed_stride_n = Eb * v_evblock_stride;
  const int64_t v_packed_stride_b = p.N * v_packed_stride_n;
  const int64_t v_packed_stride_s = 8;

  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;

  FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kMain);
  if (path_a) {
    run_path_collapse3<MK, float, true, kHasMask, kCausal>(
        q_ptr, k_packed_ptr, v_packed_ptr, p, ts,
        q_stride.b, q_stride.n, q_stride.t,
        k_packed_stride_b, k_packed_stride_n, k_packed_stride_s,
        v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
        v_evblock_stride,
        m_stride_b, m_stride_n, m_stride_l,
        out_stride.b, out_stride.n, out_stride.t);
  } else {
    const int64_t num_groups = std::max<int64_t>(1, total_threads);
    run_path_taskloop<MK, float, true, kHasMask, kCausal>(
        q_ptr, k_packed_ptr, v_packed_ptr, p, ts, static_cast<int>(num_groups),
        q_stride.b, q_stride.n, q_stride.t,
        k_packed_stride_b, k_packed_stride_n, k_packed_stride_s,
        v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
        v_evblock_stride,
        m_stride_b, m_stride_n, m_stride_l,
        out_stride.b, out_stride.n, out_stride.t);
  }
}

template <typename MK, bool kCausal, bool kHasMask>
void run_fp32_packk_path_per_head(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    const SdpaParams& p,
    const TileSizes& ts,
    bool path_a,
    int total_threads,
    ElementStrides q_stride,
    ElementStrides k_stride,
    ElementStrides v_stride,
    ElementStrides out_stride) {
  const int64_t S_blocks = ceil_div8_i64(p.S);
  const int64_t Eb = p.Ev / 8;

  // Reused across calls (grow-only): the pack functions below overwrite every
  // element they read (including zero-padded tails), so stale contents are
  // never observed. thread_local keeps one buffer per worker; the OMP pack
  // writes disjoint regions through the shared data() pointer.
  static thread_local AlignedVector<float> v_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
    v_packed.resize(static_cast<size_t>(Eb * p.S * 8));
  }

  static thread_local AlignedVector<float> k_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
    k_packed.resize(static_cast<size_t>(S_blocks * p.E * 8));
  }

  for (int64_t b = 0; b < p.B; ++b) {
    for (int64_t n = 0; n < p.N; ++n) {
      const float* q_head = q_ptr + b * q_stride.b + n * q_stride.n;
      const float* k_head = k_ptr + b * k_stride.b + n * k_stride.n;
      const float* v_head = v_ptr + b * v_stride.b + n * v_stride.n;

      {
        FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
        pack_v_to_evblock8_one_head_strided(
            v_head, v_packed.data(), p.S, p.Ev, v_stride);
      }
      {
        FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
        pack_k_fp32_to_sblock8_one_head_strided(
            k_head, k_packed.data(), p.S, p.E, k_stride, p.scale_f);
      }

      SdpaParams p_head = p;
      p_head.B = 1;
      p_head.N = 1;
      p_head.q_ptr = q_head;
      p_head.k_ptr = k_head;
      p_head.v_ptr = v_head;
      p_head.out_ptr =
          p.out_ptr + b * out_stride.b + n * out_stride.n;

      if (p.mask_ptr != nullptr) {
        p_head.mask_ptr =
            p.mask_ptr + (b * p.N + n) * p.L * p.S;
      }
      if (p.mask_f16_ptr != nullptr) {
        const int64_t mb = b % p.mask_f16_ne3;
        const int64_t mh = n % p.mask_f16_ne2;
        p_head.mask_f16_ptr = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const char*>(p.mask_f16_ptr) +
            mh * p.mask_f16_nb2 + mb * p.mask_f16_nb3);
        p_head.mask_f16_ne2 = 1;
        p_head.mask_f16_ne3 = 1;
      }
      if (p.mask_f32_ptr != nullptr) {
        const int64_t mb = b % p.mask_f32_ne3;
        const int64_t mh = n % p.mask_f32_ne2;
        p_head.mask_f32_ptr = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(p.mask_f32_ptr) +
            mh * p.mask_f32_nb2 + mb * p.mask_f32_nb3);
        p_head.mask_f32_ne2 = 1;
        p_head.mask_f32_ne3 = 1;
      }
      if (p.max_logits_ptr != nullptr) {
        p_head.max_logits_ptr =
            p.max_logits_ptr + (b * p.N + n) * p.L;
      }
      if (p.lse_ptr != nullptr) {
        p_head.lse_ptr = p.lse_ptr + (b * p.N + n) * p.L;
      }

      run_fp32_packk_path<MK, kCausal, kHasMask>(
          q_head, k_packed.data(), v_packed.data(), p_head, ts, path_a,
          total_threads,
          ElementStrides{0, 0, q_stride.t, q_stride.d},
          ElementStrides{0, 0, out_stride.t, out_stride.d});
    }
  }
}

template <bool kCausal, bool kHasMask>
void run_selected_fp32_packk_path(
    const float* q_ptr,
    const float* k_packed_ptr,
    const float* v_packed_ptr,
    const SdpaParams& p,
    const TileSizes& ts,
    bool path_a,
    int total_threads,
    ElementStrides q_stride,
    ElementStrides out_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_SVE
  if (use_sve_microkernels()) {
    run_fp32_packk_path<MK_Fp32PackK8PQuadSve, kCausal, kHasMask>(
        q_ptr, k_packed_ptr, v_packed_ptr, p, ts, path_a, total_threads,
        q_stride, out_stride);
    return;
  }
#endif
  run_fp32_packk_path<MK_Fp32PackK8PQuad, kCausal, kHasMask>(
      q_ptr, k_packed_ptr, v_packed_ptr, p, ts, path_a, total_threads,
      q_stride, out_stride);
}

template <bool kCausal, bool kHasMask>
void run_selected_fp32_packk_path_per_head(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    const SdpaParams& p,
    const TileSizes& ts,
    bool path_a,
    int total_threads,
    ElementStrides q_stride,
    ElementStrides k_stride,
    ElementStrides v_stride,
    ElementStrides out_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_SVE
  if (use_sve_microkernels()) {
    run_fp32_packk_path_per_head<MK_Fp32PackK8PQuadSve, kCausal, kHasMask>(
        q_ptr, k_ptr, v_ptr, p, ts, path_a, total_threads,
        q_stride, k_stride, v_stride, out_stride);
    return;
  }
#endif
  run_fp32_packk_path_per_head<MK_Fp32PackK8PQuad, kCausal, kHasMask>(
      q_ptr, k_ptr, v_ptr, p, ts, path_a, total_threads,
      q_stride, k_stride, v_stride, out_stride);
}

const char* selected_mk_name() {
#if FUSED_CPP_SDPA_CACHE_HAS_SVE
  if (use_sve_microkernels()) {
    return MK_Fp32PackK8PQuadSve::kName;
  }
#endif
  return MK_Fp32PackK8PQuad::kName;
}

int64_t byte_stride_to_float_elems(int64_t byte_stride) {
  if ((byte_stride % static_cast<int64_t>(sizeof(float))) != 0) {
    throw std::invalid_argument("byte stride must be divisible by sizeof(float)");
  }
  return byte_stride / static_cast<int64_t>(sizeof(float));
}

}  // namespace

void sdpa_fp32_packqkv_pbf16pv_strided_impl(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const Config& cfg_in,
    ElementStrides q_stride,
    ElementStrides k_stride,
    ElementStrides v_stride,
    ElementStrides out_stride,
    const float* mask32,
    const uint16_t* mask_f16,
    const MaskF16Layout* mask_f16_layout,
    const float* mask_f32,
    const MaskF32Layout* mask_f32_layout) {
  Config cfg = cfg_in;
  check_config(cfg);
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
    throw std::invalid_argument("q/k/v/out must be non-null");
  }
  if (q_stride.d != 1) {
    throw std::invalid_argument("fp32 packqkv path requires contiguous Q dim0");
  }
  if (out_stride.d != 1) {
    throw std::invalid_argument("fp32 packqkv path requires contiguous output dim0");
  }
  if (cfg.scale == 0.0f) {
    cfg.scale = 1.0f / std::sqrt(static_cast<float>(cfg.E));
  }

  SdpaParams p;
  p.B = cfg.B;
  p.N = cfg.N;
  p.L = cfg.L;
  p.S = cfg.S;
  p.E = cfg.E;
  p.Ev = cfg.Ev;
  p.scale_f = cfg.scale;
  p.neg_inf = kNegInf;
  p.causal_offset = cfg.causal_offset;
  p.is_causal = cfg.causal;
  p.dtype = SdpaDtype::kFloat32;
  p.q_ptr = q;
  p.k_ptr = k;
  p.v_ptr = v;
  p.mask_ptr = mask32;
  p.mask_f16_ptr = mask_f16;
  if (mask_f16 != nullptr && mask_f16_layout != nullptr) {
    p.mask_f16_ne0 = mask_f16_layout->ne0;
    p.mask_f16_ne1 = mask_f16_layout->ne1;
    p.mask_f16_ne2 = mask_f16_layout->ne2;
    p.mask_f16_ne3 = mask_f16_layout->ne3;
    p.mask_f16_nb0 = mask_f16_layout->nb0;
    p.mask_f16_nb1 = mask_f16_layout->nb1;
    p.mask_f16_nb2 = mask_f16_layout->nb2;
    p.mask_f16_nb3 = mask_f16_layout->nb3;
  }
  p.mask_f32_ptr = mask_f32;
  if (mask_f32 != nullptr && mask_f32_layout != nullptr) {
    p.mask_f32_ne0 = mask_f32_layout->ne0;
    p.mask_f32_ne1 = mask_f32_layout->ne1;
    p.mask_f32_ne2 = mask_f32_layout->ne2;
    p.mask_f32_ne3 = mask_f32_layout->ne3;
    p.mask_f32_nb0 = mask_f32_layout->nb0;
    p.mask_f32_nb1 = mask_f32_layout->nb1;
    p.mask_f32_nb2 = mask_f32_layout->nb2;
    p.mask_f32_nb3 = mask_f32_layout->nb3;
  }
  p.out_ptr = out;

  const bool profile_on = ::fused_cpp::sdpa_profile::enabled();
  if (profile_on) {
    ::fused_cpp::sdpa_profile::reset();
  }
  const uint64_t profile_total_t0 =
      profile_on ? ::fused_cpp::sdpa_profile::now_ns() : 0;

  TileSizes ts = compute_tile_sizes_l3kv(
      cfg.B, cfg.N, cfg.S, cfg.L, cfg.E, cfg.Ev, sizeof(float));
  if (cfg.s_tile > 0) {
    ts.Sc_l2 = cfg.s_tile;
    ts.Sc_l3 = std::max<int64_t>(ts.Sc_l2, std::min<int64_t>(cfg.S, ts.Sc_l3));
    ts.Sc_l3 = (ts.Sc_l3 / ts.Sc_l2) * ts.Sc_l2;
    if (ts.Sc_l3 < ts.Sc_l2) {
      ts.Sc_l3 = ts.Sc_l2;
    }
    if (ts.Sc_l3 > cfg.S) {
      ts.Sc_l3 = cfg.S;
    }
  }

  const int64_t kv_bytes_per_bn =
      cfg.S * (cfg.E + cfg.Ev) * static_cast<int64_t>(sizeof(float));
  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l3_budget =
      static_cast<int64_t>(cache_bytes[2] * FUSED_CPP_SDPA_L3_RATIO);
  const bool kv_fits_l3 = kv_bytes_per_bn <= l3_budget;
  const int total_threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif
  const bool path_a = kv_fits_l3 || total_threads <= 1;

  const bool has_mask =
      mask32 != nullptr || mask_f16 != nullptr || mask_f32 != nullptr;
  const bool pack_per_head =
      env_flag_enabled("FUSED_CPP_SDPA_PACK_PER_HEAD");
  if (pack_per_head) {
    if (cfg.causal) {
      if (has_mask) {
        run_selected_fp32_packk_path_per_head<true, true>(
            q, k, v, p, ts, path_a, total_threads,
            q_stride, k_stride, v_stride, out_stride);
      } else {
        run_selected_fp32_packk_path_per_head<true, false>(
            q, k, v, p, ts, path_a, total_threads,
            q_stride, k_stride, v_stride, out_stride);
      }
    } else {
      if (has_mask) {
        run_selected_fp32_packk_path_per_head<false, true>(
            q, k, v, p, ts, path_a, total_threads,
            q_stride, k_stride, v_stride, out_stride);
      } else {
        run_selected_fp32_packk_path_per_head<false, false>(
            q, k, v, p, ts, path_a, total_threads,
            q_stride, k_stride, v_stride, out_stride);
      }
    }
  } else {
    const int64_t eb = cfg.Ev / 8;
    AlignedVector<float> v_packed;
    {
      FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
      v_packed.resize(static_cast<size_t>(cfg.B * cfg.N * eb * cfg.S * 8));
    }
    {
      FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
      pack_v_to_evblock8_strided(
          v, v_packed.data(), cfg.B, cfg.N, cfg.S, cfg.Ev, v_stride);
    }

    AlignedVector<float> k_packed;
    {
      FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
      k_packed.resize(static_cast<size_t>(
          cfg.B * cfg.N * ceil_div8_i64(cfg.S) * cfg.E * 8));
    }
    {
      FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
      pack_k_fp32_to_sblock8_strided(
          k, k_packed.data(), cfg.B, cfg.N, cfg.S, cfg.E, k_stride, cfg.scale);
    }

    if (cfg.causal) {
      if (has_mask) {
        run_selected_fp32_packk_path<true, true>(
            q, k_packed.data(), v_packed.data(), p, ts, path_a, total_threads,
            q_stride, out_stride);
      } else {
        run_selected_fp32_packk_path<true, false>(
            q, k_packed.data(), v_packed.data(), p, ts, path_a, total_threads,
            q_stride, out_stride);
      }
    } else {
      if (has_mask) {
        run_selected_fp32_packk_path<false, true>(
            q, k_packed.data(), v_packed.data(), p, ts, path_a, total_threads,
            q_stride, out_stride);
      } else {
        run_selected_fp32_packk_path<false, false>(
            q, k_packed.data(), v_packed.data(), p, ts, path_a, total_threads,
            q_stride, out_stride);
      }
    }
  }

  if (profile_on) {
    ::fused_cpp::sdpa_profile::add(
        ::fused_cpp::sdpa_profile::Slot::kTotal,
        ::fused_cpp::sdpa_profile::now_ns() - profile_total_t0);
    ::fused_cpp::sdpa_profile::print_summary(
        "fp32_packqkv_standalone",
        selected_mk_name(),
        p,
        pack_per_head ? (path_a ? "A-per-head" : "B-per-head")
                      : (path_a ? "A" : "B"));
  }
}

void sdpa_fp32_packqkv_pbf16pv(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const Config& cfg) {
  sdpa_fp32_packqkv_pbf16pv_strided_impl(
      q,
      k,
      v,
      out,
      cfg,
      ElementStrides{cfg.N * cfg.L * cfg.E, cfg.L * cfg.E, cfg.E, 1},
      ElementStrides{cfg.N * cfg.S * cfg.E, cfg.S * cfg.E, cfg.E, 1},
      ElementStrides{cfg.N * cfg.S * cfg.Ev, cfg.S * cfg.Ev, cfg.Ev, 1},
      ElementStrides{cfg.N * cfg.L * cfg.Ev, cfg.L * cfg.Ev, cfg.Ev, 1},
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

void reference_sdpa_fp32_mask(
    const float* q,
    const float* k,
    const float* v,
    const float* mask32,
    float* out,
    const Config& cfg_in) {
  Config cfg = cfg_in;
  check_config(cfg);
  if (cfg.scale == 0.0f) {
    cfg.scale = 1.0f / std::sqrt(static_cast<float>(cfg.E));
  }

  const int64_t q_b_stride = cfg.N * cfg.L * cfg.E;
  const int64_t q_n_stride = cfg.L * cfg.E;
  const int64_t k_b_stride = cfg.N * cfg.S * cfg.E;
  const int64_t k_n_stride = cfg.S * cfg.E;
  const int64_t v_b_stride = cfg.N * cfg.S * cfg.Ev;
  const int64_t v_n_stride = cfg.S * cfg.Ev;
  const int64_t out_b_stride = cfg.N * cfg.L * cfg.Ev;
  const int64_t out_n_stride = cfg.L * cfg.Ev;

  AlignedVector<float> scores(static_cast<size_t>(cfg.S));
  for (int64_t b = 0; b < cfg.B; ++b) {
    for (int64_t n = 0; n < cfg.N; ++n) {
      for (int64_t l = 0; l < cfg.L; ++l) {
        const float* q_row = q + b * q_b_stride + n * q_n_stride + l * cfg.E;
        const float* k_bn = k + b * k_b_stride + n * k_n_stride;
        const float* v_bn = v + b * v_b_stride + n * v_n_stride;
        float* out_row = out + b * out_b_stride + n * out_n_stride + l * cfg.Ev;

        const int64_t visible = cfg.causal
            ? clamp_i64(l + cfg.causal_offset + 1, 0, cfg.S)
            : cfg.S;

        float m = kNegInf;
        for (int64_t s = 0; s < visible; ++s) {
          const float* k_row = k_bn + s * cfg.E;
          float acc = 0.0f;
          for (int64_t e = 0; e < cfg.E; ++e) {
            acc += q_row[e] * k_row[e];
          }
          scores[static_cast<size_t>(s)] = acc * cfg.scale;
          if (mask32 != nullptr) {
            scores[static_cast<size_t>(s)] +=
                mask32[((b * cfg.N + n) * cfg.L + l) * cfg.S + s];
          }
          m = std::max(m, scores[static_cast<size_t>(s)]);
        }

        std::fill(out_row, out_row + cfg.Ev, 0.0f);
        if (visible <= 0) {
          continue;
        }

        float denom = 0.0f;
        for (int64_t s = 0; s < visible; ++s) {
          const float prob = std::exp(scores[static_cast<size_t>(s)] - m);
          denom += prob;
          const float* v_row = v_bn + s * cfg.Ev;
          for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
            out_row[ev] += prob * v_row[ev];
          }
        }
        const float inv = 1.0f / denom;
        for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
          out_row[ev] *= inv;
        }
      }
    }
  }
}

void reference_sdpa_fp32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const Config& cfg) {
  reference_sdpa_fp32_mask(q, k, v, nullptr, out, cfg);
}

double counted_gflops(const Config& cfg, double mean_ms) {
  int64_t active_per_head = 0;
  if (!cfg.causal) {
    active_per_head = cfg.L * cfg.S;
  } else {
    for (int64_t l = 0; l < cfg.L; ++l) {
      active_per_head += clamp_i64(l + cfg.causal_offset + 1, 0, cfg.S);
    }
  }
  const double active =
      static_cast<double>(cfg.B) * static_cast<double>(cfg.N) *
      static_cast<double>(active_per_head);
  const double flops_per_score =
      2.0 * static_cast<double>(cfg.E) +
      2.0 * static_cast<double>(cfg.Ev) + 5.0;
  return active * flops_per_score / (mean_ms * 1.0e6);
}

double checksum(const float* data, int64_t size) {
  double sum = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    sum += static_cast<double>(data[i]);
  }
  return sum;
}

double max_abs_diff(const float* a, const float* b, int64_t size) {
  double m = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    m = std::max(m, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return m;
}

int run_llamacpp_strided_impl(
    const float* q,
    const float* k,
    const float* v,
    const uint16_t* mask_f16,
    const float* mask_f32,
    float* out,
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
    const MaskF16Layout& mask_layout,
    float scale) {
  const bool use_f16_mask = mask_f16 != nullptr;
  const bool use_f32_mask = mask_f32 != nullptr;
  if (use_f16_mask && use_f32_mask) {
    return 2;
  }
  const bool use_mask = use_f16_mask || use_f32_mask;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      (use_mask && mask_f16 == nullptr && mask_f32 == nullptr)) {
    return 1;
  }
  if (B <= 0 || H <= 0 || L <= 0 || S <= 0 || D <= 0 || DV <= 0) {
    return 2;
  }
  if ((DV % 8) != 0) {
    return 3;
  }
  const int64_t q_e = byte_stride_to_float_elems(q_nb0);
  const int64_t q_l = byte_stride_to_float_elems(q_nb1);
  const int64_t q_h = byte_stride_to_float_elems(q_nb2);
  const int64_t q_b = byte_stride_to_float_elems(q_nb3);
  const int64_t k_e = byte_stride_to_float_elems(k_nb0);
  const int64_t k_s = byte_stride_to_float_elems(k_nb1);
  const int64_t k_h = byte_stride_to_float_elems(k_nb2);
  const int64_t k_b = byte_stride_to_float_elems(k_nb3);
  const int64_t v_e = byte_stride_to_float_elems(v_nb0);
  const int64_t v_s = byte_stride_to_float_elems(v_nb1);
  const int64_t v_h = byte_stride_to_float_elems(v_nb2);
  const int64_t v_b = byte_stride_to_float_elems(v_nb3);
  const int64_t o_e = byte_stride_to_float_elems(o_nb0);
  const int64_t o_h = byte_stride_to_float_elems(o_nb1);
  const int64_t o_l = byte_stride_to_float_elems(o_nb2);
  const int64_t o_b = byte_stride_to_float_elems(o_nb3);
  if (q_e != 1 || o_e != 1) {
    return 3;
  }

  if (use_mask) {
    check_mask_layout(L, S, mask_layout, use_f16_mask ? "F16" : "F32");
  }

  Config cfg;
  cfg.B = B;
  cfg.N = H;
  cfg.L = L;
  cfg.S = S;
  cfg.E = D;
  cfg.Ev = DV;
  cfg.causal = false;
  cfg.scale = scale;
  cfg.causal_offset = S - L;
  sdpa_fp32_packqkv_pbf16pv_strided_impl(
      q,
      k,
      v,
      out,
      cfg,
      ElementStrides{q_b, q_h, q_l, q_e},
      ElementStrides{k_b, k_h, k_s, k_e},
      ElementStrides{v_b, v_h, v_s, v_e},
      ElementStrides{o_b, o_h, o_l, o_e},
      nullptr,
      mask_f16,
      use_f16_mask ? &mask_layout : nullptr,
      mask_f32,
      use_f32_mask ? &mask_layout : nullptr);

  return 0;
}

}  // namespace fp32_packqkv_sdpa

extern "C" FUSED_CPP_FP32_PACKQKV_API int
fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_contiguous(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t B,
    int64_t N,
    int64_t L,
    int64_t S,
    int64_t E,
    int64_t Ev,
    int causal,
    float scale) {
  try {
    fp32_packqkv_sdpa::Config cfg;
    cfg.B = B;
    cfg.N = N;
    cfg.L = L;
    cfg.S = S;
    cfg.E = E;
    cfg.Ev = Ev;
    cfg.causal = causal != 0;
    cfg.scale = scale;
    cfg.causal_offset = S - L;
    fp32_packqkv_sdpa::sdpa_fp32_packqkv_pbf16pv(q, k, v, out, cfg);
    return 0;
  } catch (const std::invalid_argument&) {
    return 2;
  } catch (const std::exception&) {
    return 100;
  } catch (...) {
    return 101;
  }
}

extern "C" FUSED_CPP_FP32_PACKQKV_API int
fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp(
    const float* q,
    const float* k,
    const float* v,
    float* out,
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
    float scale) {
  try {
    return fp32_packqkv_sdpa::run_llamacpp_strided_impl(
        q,
        k,
        v,
        nullptr,
        nullptr,
        out,
        B,
        H,
        L,
        S,
        D,
        DV,
        q_nb0,
        q_nb1,
        q_nb2,
        q_nb3,
        k_nb0,
        k_nb1,
        k_nb2,
        k_nb3,
        v_nb0,
        v_nb1,
        v_nb2,
        v_nb3,
        o_nb0,
        o_nb1,
        o_nb2,
        o_nb3,
        fp32_packqkv_sdpa::MaskF16Layout{},
        scale);
  } catch (const std::invalid_argument&) {
    return 2;
  } catch (const std::exception&) {
    return 100;
  } catch (...) {
    return 101;
  }
}

extern "C" FUSED_CPP_FP32_PACKQKV_API int
fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f16(
    const float* q,
    const float* k,
    const float* v,
    const uint16_t* mask,
    float* out,
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
    int64_t mask_ne0,
    int64_t mask_ne1,
    int64_t mask_ne2,
    int64_t mask_ne3,
    int64_t mask_nb0,
    int64_t mask_nb1,
    int64_t mask_nb2,
    int64_t mask_nb3,
    float scale) {
  try {
    if (mask == nullptr) {
      return 1;
    }
    return fp32_packqkv_sdpa::run_llamacpp_strided_impl(
        q,
        k,
        v,
        mask,
        nullptr,
        out,
        B,
        H,
        L,
        S,
        D,
        DV,
        q_nb0,
        q_nb1,
        q_nb2,
        q_nb3,
        k_nb0,
        k_nb1,
        k_nb2,
        k_nb3,
        v_nb0,
        v_nb1,
        v_nb2,
        v_nb3,
        o_nb0,
        o_nb1,
        o_nb2,
        o_nb3,
        fp32_packqkv_sdpa::MaskF16Layout{
            mask_ne0,
            mask_ne1,
            mask_ne2,
            mask_ne3,
            mask_nb0,
            mask_nb1,
            mask_nb2,
            mask_nb3},
        scale);
  } catch (const std::invalid_argument&) {
    return 2;
  } catch (const std::exception&) {
    return 100;
  } catch (...) {
    return 101;
  }
}

extern "C" FUSED_CPP_FP32_PACKQKV_API int
fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    float* out,
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
    int64_t mask_ne0,
    int64_t mask_ne1,
    int64_t mask_ne2,
    int64_t mask_ne3,
    int64_t mask_nb0,
    int64_t mask_nb1,
    int64_t mask_nb2,
    int64_t mask_nb3,
    float scale) {
  try {
    if (mask == nullptr) {
      return 1;
    }
    return fp32_packqkv_sdpa::run_llamacpp_strided_impl(
        q,
        k,
        v,
        nullptr,
        mask,
        out,
        B,
        H,
        L,
        S,
        D,
        DV,
        q_nb0,
        q_nb1,
        q_nb2,
        q_nb3,
        k_nb0,
        k_nb1,
        k_nb2,
        k_nb3,
        v_nb0,
        v_nb1,
        v_nb2,
        v_nb3,
        o_nb0,
        o_nb1,
        o_nb2,
        o_nb3,
        fp32_packqkv_sdpa::MaskF32Layout{
            mask_ne0,
            mask_ne1,
            mask_ne2,
            mask_ne3,
            mask_nb0,
            mask_nb1,
            mask_nb2,
            mask_nb3},
        scale);
  } catch (const std::invalid_argument&) {
    return 2;
  } catch (const std::exception&) {
    return 100;
  } catch (...) {
    return 101;
  }
}
