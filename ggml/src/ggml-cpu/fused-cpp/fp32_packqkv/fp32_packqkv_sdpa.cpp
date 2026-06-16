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


#include "sdpa_common.h"
#include "sdpa_tile_sizes.h"
#include "sdpa_pack_utils.h"
#include "sdpa_flash2_neon_l3kv_impl.h"

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

constexpr int64_t kEvBlock = 4;
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

inline int64_t ceil_div4_i64(int64_t v) {
  return (v + 3) / 4;
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

inline void check_config(const Config& cfg) {
  if (cfg.B <= 0 || cfg.N <= 0 || cfg.L <= 0 || cfg.S <= 0 ||
      cfg.E <= 0 || cfg.Ev <= 0) {
    throw std::invalid_argument("shape dimensions must be positive");
  }
  if (cfg.Ev % kEvBlock != 0) {
    throw std::invalid_argument("fp32 packqkv path requires Ev % 4 == 0");
  }
  if (cfg.s_tile < 0 || cfg.s_tile % 4 != 0) {
    throw std::invalid_argument(
        "fp32 packqkv path requires s_tile == 0 or s_tile % 4 == 0");
  }
}

void pack_k_fp32_to_sblock4_one_head_strided(
    const float* k_src,
    float* k_dst,
    int64_t S,
    int64_t E,
    ElementStrides k_stride,
    float scale) {
  const int64_t S_blocks = ceil_div4_i64(S);
  const int64_t dst_stride_sb = E * 4;

  for (int64_t sb = 0; sb < S_blocks; ++sb) {
    const int64_t s0 = sb * 4;
    const int64_t remain = S - s0;
    const float* src = k_src + s0 * k_stride.t;
    float* dst = k_dst + sb * dst_stride_sb;
    for (int64_t e = 0; e < E; ++e) {
      float* d = dst + e * 4;
      const int64_t ed = e * k_stride.d;
      if (remain >= 4) {
        d[0] = src[0 * k_stride.t + ed] * scale;
        d[1] = src[1 * k_stride.t + ed] * scale;
        d[2] = src[2 * k_stride.t + ed] * scale;
        d[3] = src[3 * k_stride.t + ed] * scale;
      } else {
        const int64_t tail = std::max<int64_t>(0, remain);
        int64_t j = 0;
        for (; j < tail; ++j) {
          d[j] = src[j * k_stride.t + ed] * scale;
        }
        for (; j < 4; ++j) {
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
  const int64_t S_blocks = ceil_div4_i64(S);
  const int64_t dst_stride_b = N * S_blocks * E * 4;
  const int64_t dst_stride_n = S_blocks * E * 4;
  const int64_t dst_stride_sb = E * 4;

#ifdef _OPENMP
  #pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t sb = 0; sb < S_blocks; ++sb) {
        const int64_t s0 = sb * 4;
        const int64_t remain = S - s0;
        const float* src = k_src + b * k_stride.b
                                 + n * k_stride.n
                                 + s0 * k_stride.t;
        float* dst = k_dst + b * dst_stride_b
                           + n * dst_stride_n
                           + sb * dst_stride_sb;
        for (int64_t e = 0; e < E; ++e) {
          float* d = dst + e * 4;
          const int64_t ed = e * k_stride.d;
          if (remain >= 4) {
            d[0] = src[0 * k_stride.t + ed] * scale;
            d[1] = src[1 * k_stride.t + ed] * scale;
            d[2] = src[2 * k_stride.t + ed] * scale;
            d[3] = src[3 * k_stride.t + ed] * scale;
          } else {
            const int64_t tail = std::max<int64_t>(0, remain);
            int64_t j = 0;
            for (; j < tail; ++j) {
              d[j] = src[j * k_stride.t + ed] * scale;
            }
            for (; j < 4; ++j) {
              d[j] = 0.0f;
            }
          }
        }
      }
    }
  }
}

void pack_v_to_evblock4_one_head_strided(
    const float* v_src,
    float* v_dst,
    int64_t S,
    int64_t Ev,
    ElementStrides v_stride) {
  const int64_t Eb = Ev / 4;
  const int64_t evblock_stride_dst = S * 4;

  for (int64_t eb = 0; eb < Eb; ++eb) {
    const int64_t ev = eb * 4;
    const float* src = v_src + ev * v_stride.d;
    float* dst = v_dst + eb * evblock_stride_dst;
    for (int64_t s = 0; s < S; ++s) {
      const float* row = src + s * v_stride.t;
      float* packed = dst + s * 4;
      if (v_stride.d == 1) {
        packed[0] = row[0];
        packed[1] = row[1];
        packed[2] = row[2];
        packed[3] = row[3];
      } else {
        for (int64_t i = 0; i < 4; ++i) {
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
  const int64_t Eb = Ev / 4;
  const int64_t dst_stride_b = N * Eb * S * 4;
  const int64_t dst_stride_n = Eb * S * 4;
  const int64_t evblock_stride_dst = S * 4;

#ifdef _OPENMP
  #pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t eb = 0; eb < Eb; ++eb) {
        const int64_t ev = eb * 4;
        const float* src = v_src + b * v_stride.b
                                 + n * v_stride.n
                                 + ev * v_stride.d;
        float* dst = v_dst + b * dst_stride_b
                           + n * dst_stride_n
                           + eb * evblock_stride_dst;
        for (int64_t s = 0; s < S; ++s) {
          const float* row = src + s * v_stride.t;
          float* packed = dst + s * 4;
          if (v_stride.d == 1) {
            packed[0] = row[0];
            packed[1] = row[1];
            packed[2] = row[2];
            packed[3] = row[3];
          } else {
            for (int64_t i = 0; i < 4; ++i) {
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

struct MK_Fp32PackK8PQuad {
  static constexpr const char* kName = "fp32_packk8_pquad";
  static constexpr bool kHasKSBlock8Layout = true;
  static constexpr bool kHasKSBlock4Layout = true;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  static constexpr bool kSupportsQktNeonRowMax = true;
#endif

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  // 4x4 register-blocked GEMM compute (C = A.B). a_i = row i's 4 reduction
  // values (laneq broadcast); b_j = the j-th reduction value's 4 columns;
  // c_i = row i's 4-column accumulator.
  static inline void mm4x4_seed(
      float32x4_t& c0, float32x4_t& c1, float32x4_t& c2, float32x4_t& c3,
      float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3,
      float32x4_t b0, float32x4_t b1, float32x4_t b2, float32x4_t b3) {
    c0 = vmulq_laneq_f32(b0, a0, 0); c1 = vmulq_laneq_f32(b0, a1, 0);
    c2 = vmulq_laneq_f32(b0, a2, 0); c3 = vmulq_laneq_f32(b0, a3, 0);
    c0 = vfmaq_laneq_f32(c0, b1, a0, 1); c1 = vfmaq_laneq_f32(c1, b1, a1, 1);
    c2 = vfmaq_laneq_f32(c2, b1, a2, 1); c3 = vfmaq_laneq_f32(c3, b1, a3, 1);
    c0 = vfmaq_laneq_f32(c0, b2, a0, 2); c1 = vfmaq_laneq_f32(c1, b2, a1, 2);
    c2 = vfmaq_laneq_f32(c2, b2, a2, 2); c3 = vfmaq_laneq_f32(c3, b2, a3, 2);
    c0 = vfmaq_laneq_f32(c0, b3, a0, 3); c1 = vfmaq_laneq_f32(c1, b3, a1, 3);
    c2 = vfmaq_laneq_f32(c2, b3, a2, 3); c3 = vfmaq_laneq_f32(c3, b3, a3, 3);
  }
  static inline void mm4x4_fmla(
      float32x4_t& c0, float32x4_t& c1, float32x4_t& c2, float32x4_t& c3,
      float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3,
      float32x4_t b0, float32x4_t b1, float32x4_t b2, float32x4_t b3) {
    c0 = vfmaq_laneq_f32(c0, b0, a0, 0); c1 = vfmaq_laneq_f32(c1, b0, a1, 0);
    c2 = vfmaq_laneq_f32(c2, b0, a2, 0); c3 = vfmaq_laneq_f32(c3, b0, a3, 0);
    c0 = vfmaq_laneq_f32(c0, b1, a0, 1); c1 = vfmaq_laneq_f32(c1, b1, a1, 1);
    c2 = vfmaq_laneq_f32(c2, b1, a2, 1); c3 = vfmaq_laneq_f32(c3, b1, a3, 1);
    c0 = vfmaq_laneq_f32(c0, b2, a0, 2); c1 = vfmaq_laneq_f32(c1, b2, a1, 2);
    c2 = vfmaq_laneq_f32(c2, b2, a2, 2); c3 = vfmaq_laneq_f32(c3, b2, a3, 2);
    c0 = vfmaq_laneq_f32(c0, b3, a0, 3); c1 = vfmaq_laneq_f32(c1, b3, a1, 3);
    c2 = vfmaq_laneq_f32(c2, b3, a2, 3); c3 = vfmaq_laneq_f32(c3, b3, a3, 3);
  }
#endif

  // Scalar QKT for a partial tile (Lq<4 rows or Sk<4 cols). K is sblock4
  // (Kp[e*4 + col]); scale already folded into packed K.
  static inline void qkt_4x4_tail(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    (void)scale;
    for (int i = 0; i < Lq; ++i) {
      const float* q = Q + i * q_row_stride;
      float* out = scores_buf + i * scores_row_stride;
      for (int j = 0; j < Sk; ++j) {
        float sum = 0.0f;
        for (int64_t e = 0; e < E; ++e) {
          sum += q[e] * Kp[e * 4 + j];
        }
        out[j] = sum;
      }
    }
  }

  // C[4x4] = Q[4xE] . K^T over E. A=Q rows (laneq over 4 e-values), B=K from
  // sblock4 (Kp + e*4, 4 cols). 4 A + 4 B + 4 C = 12 vector regs; a second
  // A/B set (X/Y) double-buffers the load/compute pipeline (load0, load1,
  // compute0, load2, compute1, ...). C is mul-seeded on the first e-block
  // (no zero-init), fmla after; scalar tail for E % 4.
  static inline void qkt_4x4(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    (void)scale;  // folded into packed K
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;

    float32x4_t c0, c1, c2, c3;
    const int64_t nfull = (E / 4) * 4;

    float32x4_t qa0, qa1, qa2, qa3, ka0, ka1, ka2, ka3;  // buffer 0
    float32x4_t qb0, qb1, qb2, qb3, kb0, kb1, kb2, kb3;  // buffer 1
#define QKT4_LOAD0(BLK)                                                     \
    do { const int64_t e_ = (BLK) * 4;                                      \
      qa0 = vld1q_f32(q0 + e_); qa1 = vld1q_f32(q1 + e_);                   \
      qa2 = vld1q_f32(q2 + e_); qa3 = vld1q_f32(q3 + e_);                   \
      ka0 = vld1q_f32(Kp + e_ * 4 + 0);  ka1 = vld1q_f32(Kp + e_ * 4 + 4);  \
      ka2 = vld1q_f32(Kp + e_ * 4 + 8);  ka3 = vld1q_f32(Kp + e_ * 4 + 12); \
    } while (0)
#define QKT4_LOAD1(BLK)                                                     \
    do { const int64_t e_ = (BLK) * 4;                                      \
      qb0 = vld1q_f32(q0 + e_); qb1 = vld1q_f32(q1 + e_);                   \
      qb2 = vld1q_f32(q2 + e_); qb3 = vld1q_f32(q3 + e_);                   \
      kb0 = vld1q_f32(Kp + e_ * 4 + 0);  kb1 = vld1q_f32(Kp + e_ * 4 + 4);  \
      kb2 = vld1q_f32(Kp + e_ * 4 + 8);  kb3 = vld1q_f32(Kp + e_ * 4 + 12); \
    } while (0)

    const int64_t nb = nfull / 4;
    bool seeded = false;
    if (nb > 0) {
      // prologue: load0, load1, compute0* (mul-seed; no zero-init)
      QKT4_LOAD0(0);
      if (nb >= 2) QKT4_LOAD1(1);
      mm4x4_seed(c0, c1, c2, c3, qa0, qa1, qa2, qa3, ka0, ka1, ka2, ka3);
      seeded = true;
      // steady: load0, compute1, load1, compute0 (load one block ahead)
      int64_t k = 2;
      for (; k + 1 < nb; k += 2) {
        QKT4_LOAD0(k);
        mm4x4_fmla(c0, c1, c2, c3, qb0, qb1, qb2, qb3, kb0, kb1, kb2, kb3);
        QKT4_LOAD1(k + 1);
        mm4x4_fmla(c0, c1, c2, c3, qa0, qa1, qa2, qa3, ka0, ka1, ka2, ka3);
      }
      // epilogue: drain (two blocks remain when k<nb, else the last compute1)
      if (nb >= 2) {
        if (k < nb) {
          QKT4_LOAD0(k);
          mm4x4_fmla(c0, c1, c2, c3, qb0, qb1, qb2, qb3, kb0, kb1, kb2, kb3);
          mm4x4_fmla(c0, c1, c2, c3, qa0, qa1, qa2, qa3, ka0, ka1, ka2, ka3);
        } else {
          mm4x4_fmla(c0, c1, c2, c3, qb0, qb1, qb2, qb3, kb0, kb1, kb2, kb3);
        }
      }
    }
#undef QKT4_LOAD0
#undef QKT4_LOAD1

    // tail: E % 4 (and the E < 4 case); mul-seed the first if not yet seeded
    int64_t et = nfull;
    if (!seeded) {
      const float32x4_t kv = vld1q_f32(Kp + et * 4);
      c0 = vmulq_n_f32(kv, q0[et]); c1 = vmulq_n_f32(kv, q1[et]);
      c2 = vmulq_n_f32(kv, q2[et]); c3 = vmulq_n_f32(kv, q3[et]);
      ++et;
    }
    for (; et < E; ++et) {
      const float32x4_t kv = vld1q_f32(Kp + et * 4);
      c0 = vfmaq_n_f32(c0, kv, q0[et]);
      c1 = vfmaq_n_f32(c1, kv, q1[et]);
      c2 = vfmaq_n_f32(c2, kv, q2[et]);
      c3 = vfmaq_n_f32(c3, kv, q3[et]);
    }

    vst1q_f32(scores_buf + 0 * scores_row_stride, c0);
    vst1q_f32(scores_buf + 1 * scores_row_stride, c1);
    vst1q_f32(scores_buf + 2 * scores_row_stride, c2);
    vst1q_f32(scores_buf + 3 * scores_row_stride, c3);
#else
    qkt_4x4_tail(Q, q_row_stride, Kp, 0, E, scale, scores_buf,
                 scores_row_stride, 4, 4);
#endif
  }

  // Scalar PV for a partial tile (Lq<4 rows or Ev<4 cols). Accumulates into O.
  static inline void pv_4x4_tail(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    for (int i = 0; i < Lq; ++i) {
      const float* p = P_hat + i * P_row_stride;
      float* o = O + i * o_row_stride;
      for (int j = 0; j < Ev; ++j) {
        float sum = o[j];
        for (int64_t k = 0; k < Sk; ++k) {
          sum += p[k] * V[k * v_row_stride + j];
        }
        o[j] = sum;
      }
    }
  }

  // O[4 rows x 4 Ev] += P_hat[4xSk] . V[Skx4] over Sk. A=P rows (laneq over 4
  // s-values), B=V (V + s*v_row_stride, 4 Ev). Same 12-reg + double-buffer
  // pipeline as qkt_4x4. C is seeded from O (online-softmax accumulator), so
  // every block uses fmla; scalar tail for Sk % 4.
  static inline void pv_4x4(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    const float* p0 = P_hat + 0 * P_row_stride;
    const float* p1 = P_hat + 1 * P_row_stride;
    const float* p2 = P_hat + 2 * P_row_stride;
    const float* p3 = P_hat + 3 * P_row_stride;

    float32x4_t c0 = vld1q_f32(O + 0 * o_row_stride);
    float32x4_t c1 = vld1q_f32(O + 1 * o_row_stride);
    float32x4_t c2 = vld1q_f32(O + 2 * o_row_stride);
    float32x4_t c3 = vld1q_f32(O + 3 * o_row_stride);

    const int64_t nfull = (Sk / 4) * 4;

    float32x4_t pa0, pa1, pa2, pa3, va0, va1, va2, va3;  // buffer 0
    float32x4_t pb0, pb1, pb2, pb3, vb0, vb1, vb2, vb3;  // buffer 1
#define PV4_LOAD0(BLK)                                                  \
    do { const int64_t s_ = (BLK) * 4;                                  \
      pa0 = vld1q_f32(p0 + s_); pa1 = vld1q_f32(p1 + s_);               \
      pa2 = vld1q_f32(p2 + s_); pa3 = vld1q_f32(p3 + s_);               \
      va0 = vld1q_f32(V + (s_ + 0) * v_row_stride);                     \
      va1 = vld1q_f32(V + (s_ + 1) * v_row_stride);                     \
      va2 = vld1q_f32(V + (s_ + 2) * v_row_stride);                     \
      va3 = vld1q_f32(V + (s_ + 3) * v_row_stride);                     \
    } while (0)
#define PV4_LOAD1(BLK)                                                  \
    do { const int64_t s_ = (BLK) * 4;                                  \
      pb0 = vld1q_f32(p0 + s_); pb1 = vld1q_f32(p1 + s_);               \
      pb2 = vld1q_f32(p2 + s_); pb3 = vld1q_f32(p3 + s_);               \
      vb0 = vld1q_f32(V + (s_ + 0) * v_row_stride);                     \
      vb1 = vld1q_f32(V + (s_ + 1) * v_row_stride);                     \
      vb2 = vld1q_f32(V + (s_ + 2) * v_row_stride);                     \
      vb3 = vld1q_f32(V + (s_ + 3) * v_row_stride);                     \
    } while (0)

    const int64_t nb = nfull / 4;
    if (nb > 0) {
      // prologue: load0, load1, compute0 (fmla; C already seeded from O)
      PV4_LOAD0(0);
      if (nb >= 2) PV4_LOAD1(1);
      mm4x4_fmla(c0, c1, c2, c3, pa0, pa1, pa2, pa3, va0, va1, va2, va3);
      // steady: load0, compute1, load1, compute0 (load one block ahead)
      int64_t k = 2;
      for (; k + 1 < nb; k += 2) {
        PV4_LOAD0(k);
        mm4x4_fmla(c0, c1, c2, c3, pb0, pb1, pb2, pb3, vb0, vb1, vb2, vb3);
        PV4_LOAD1(k + 1);
        mm4x4_fmla(c0, c1, c2, c3, pa0, pa1, pa2, pa3, va0, va1, va2, va3);
      }
      // epilogue: drain (two blocks remain when k<nb, else the last compute1)
      if (nb >= 2) {
        if (k < nb) {
          PV4_LOAD0(k);
          mm4x4_fmla(c0, c1, c2, c3, pb0, pb1, pb2, pb3, vb0, vb1, vb2, vb3);
          mm4x4_fmla(c0, c1, c2, c3, pa0, pa1, pa2, pa3, va0, va1, va2, va3);
        } else {
          mm4x4_fmla(c0, c1, c2, c3, pb0, pb1, pb2, pb3, vb0, vb1, vb2, vb3);
        }
      }
    }
#undef PV4_LOAD0
#undef PV4_LOAD1

    // tail: Sk % 4
    for (int64_t st = nfull; st < Sk; ++st) {
      const float32x4_t vv = vld1q_f32(V + st * v_row_stride);
      c0 = vfmaq_n_f32(c0, vv, p0[st]);
      c1 = vfmaq_n_f32(c1, vv, p1[st]);
      c2 = vfmaq_n_f32(c2, vv, p2[st]);
      c3 = vfmaq_n_f32(c3, vv, p3[st]);
    }

    vst1q_f32(O + 0 * o_row_stride, c0);
    vst1q_f32(O + 1 * o_row_stride, c1);
    vst1q_f32(O + 2 * o_row_stride, c2);
    vst1q_f32(O + 3 * o_row_stride, c3);
#else
    pv_4x4_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, 4, 4);
#endif
  }

};


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
  const int64_t S_blocks = ceil_div4_i64(p.S);
  const int64_t k_packed_stride_n = S_blocks * p.E * 4;
  const int64_t k_packed_stride_b = p.N * k_packed_stride_n;
  const int64_t k_packed_stride_s = p.E;

  const int64_t Eb = p.Ev / 4;
  const int64_t v_evblock_stride = p.S * 4;
  const int64_t v_packed_stride_n = Eb * v_evblock_stride;
  const int64_t v_packed_stride_b = p.N * v_packed_stride_n;
  const int64_t v_packed_stride_s = 4;

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
  const int64_t S_blocks = ceil_div4_i64(p.S);
  const int64_t Eb = p.Ev / 4;

  // Reused across calls (grow-only): the pack functions below overwrite every
  // element they read (including zero-padded tails), so stale contents are
  // never observed. thread_local keeps one buffer per worker; the OMP pack
  // writes disjoint regions through the shared data() pointer.
  static thread_local AlignedVector<float> v_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
    v_packed.resize(static_cast<size_t>(Eb * p.S * 4));
  }

  static thread_local AlignedVector<float> k_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
    k_packed.resize(static_cast<size_t>(S_blocks * p.E * 4));
  }

  for (int64_t b = 0; b < p.B; ++b) {
    for (int64_t n = 0; n < p.N; ++n) {
      const float* q_head = q_ptr + b * q_stride.b + n * q_stride.n;
      const float* k_head = k_ptr + b * k_stride.b + n * k_stride.n;
      const float* v_head = v_ptr + b * v_stride.b + n * v_stride.n;

      {
        FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
        pack_v_to_evblock4_one_head_strided(
            v_head, v_packed.data(), p.S, p.Ev, v_stride);
      }
      {
        FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
        pack_k_fp32_to_sblock4_one_head_strided(
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
  run_fp32_packk_path_per_head<MK_Fp32PackK8PQuad, kCausal, kHasMask>(
      q_ptr, k_ptr, v_ptr, p, ts, path_a, total_threads,
      q_stride, k_stride, v_stride, out_stride);
}

const char* selected_mk_name() {
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
    const int64_t eb = cfg.Ev / 4;
    AlignedVector<float> v_packed;
    {
      FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
      v_packed.resize(static_cast<size_t>(cfg.B * cfg.N * eb * cfg.S * 4));
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
          cfg.B * cfg.N * ceil_div4_i64(cfg.S) * cfg.E * 4));
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
