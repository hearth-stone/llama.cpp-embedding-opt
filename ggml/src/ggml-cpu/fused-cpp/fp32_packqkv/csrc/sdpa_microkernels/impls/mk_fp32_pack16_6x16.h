#pragma once
// KleidiAI-style width-16 k-major packing + 6x16 stream-B microkernels for the
// fp32 packqkv SDPA path. Header-only so the packers/kernels can be unit-tested
// standalone and reached from the single fp32_packqkv_sdpa.cpp translation unit.
//
// Packed layouts (per head):
//   K (QKT RHS): [S_b16][E][16]  dst[sb*E*16 + e*16 + j] = scale * K[sb*16+j, e]
//                (e-major within an S-block of 16 keys; scale folded in)
//   V (PV  RHS): [Ev_b16][S][16] dst[evb*S*16 + s*16 + j] = V[s, evb*16+j]
//                (s-major within an Ev-block of 16; reduction dim S is contiguous)
//
// Both feed a 6x16 stream-B kernel that holds 6 LHS rows + 24 accumulators and
// streams the packed RHS through 2 NEON registers (KleidiAI 6x16 schedule).

#include <cstdint>

#include "sdpa_microkernels/neon_cache_config.h"

namespace fused_cpp {
namespace sdpa_pack16 {

#if defined(__GNUC__) || defined(__clang__)
#define FUSED_CPP_PACK16_NOINLINE __attribute__((noinline))
#else
#define FUSED_CPP_PACK16_NOINLINE
#endif

constexpr int kMR = 6;   // rows of LHS held per tile
constexpr int kNR = 16;  // cols of RHS produced per tile

inline int64_t ceil_div16(int64_t v) { return (v + 15) / 16; }

inline int64_t k_packed_size_per_head(int64_t S, int64_t E) {
  return ceil_div16(S) * E * 16;
}

inline int64_t v_packed_size_per_head(int64_t S, int64_t Ev) {
  return ceil_div16(Ev) * S * 16;
}

// Pack one head of K into [S_b16][E][16] with the QK scale folded in.
// k_t_stride: element stride between successive keys (S dim).
// k_d_stride: element stride between successive E components.
inline void pack_k_to_sblock16_one_head(
    const float* k_src,
    float* k_dst,
    int64_t S,
    int64_t E,
    int64_t k_t_stride,
    int64_t k_d_stride,
    float scale) {
  const int64_t S_b16 = ceil_div16(S);
  const int64_t dst_stride_sb = E * 16;
  for (int64_t sb = 0; sb < S_b16; ++sb) {
    const int64_t s0 = sb * 16;
    const int64_t remain = S - s0;
    const int64_t n = remain >= 16 ? 16 : (remain > 0 ? remain : 0);
    float* dst = k_dst + sb * dst_stride_sb;
    for (int64_t e = 0; e < E; ++e) {
      float* d = dst + e * 16;
      const int64_t ed = e * k_d_stride;
      int64_t j = 0;
      for (; j < n; ++j) {
        d[j] = k_src[(s0 + j) * k_t_stride + ed] * scale;
      }
      for (; j < 16; ++j) {
        d[j] = 0.0f;
      }
    }
  }
}

// Pack one head of V into [Ev_b16][S][16].
// v_t_stride: element stride between successive keys (S dim).
// v_d_stride: element stride between successive Ev components.
inline void pack_v_to_evblock16_one_head(
    const float* v_src,
    float* v_dst,
    int64_t S,
    int64_t Ev,
    int64_t v_t_stride,
    int64_t v_d_stride) {
  const int64_t Ev_b16 = ceil_div16(Ev);
  const int64_t dst_stride_evb = S * 16;
  for (int64_t evb = 0; evb < Ev_b16; ++evb) {
    const int64_t ev0 = evb * 16;
    const int64_t remain = Ev - ev0;
    const int64_t n = remain >= 16 ? 16 : (remain > 0 ? remain : 0);
    float* dst = v_dst + evb * dst_stride_evb;
    if (v_d_stride == 1 && n == 16) {
      // contiguous 16-wide Ev block: SIMD copy.
      for (int64_t s = 0; s < S; ++s) {
        const float* row = v_src + s * v_t_stride + ev0;
        float* packed = dst + s * 16;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
        vst1q_f32(packed + 0, vld1q_f32(row + 0));
        vst1q_f32(packed + 4, vld1q_f32(row + 4));
        vst1q_f32(packed + 8, vld1q_f32(row + 8));
        vst1q_f32(packed + 12, vld1q_f32(row + 12));
#else
        for (int j = 0; j < 16; ++j) packed[j] = row[j];
#endif
      }
      continue;
    }
    for (int64_t s = 0; s < S; ++s) {
      const float* row = v_src + s * v_t_stride + ev0 * v_d_stride;
      float* packed = dst + s * 16;
      int64_t j = 0;
      for (; j < n; ++j) {
        packed[j] = row[j * v_d_stride];
      }
      for (; j < 16; ++j) {
        packed[j] = 0.0f;
      }
    }
  }
}

// ── QKT 6x16 stream-B kernel ────────────────────────────────────────────────
//
// scores[R x Sk] = Q[R x E] (q_row_stride) . Kp, where Kp is one S-block of the
// packed K ([E][16], e-major). Scale is already folded into Kp. R = active rows
// (1..6), Sk = active cols (1..16). Q rows are held (lane-broadcast, unroll-4
// over E); the 16 packed K cols are streamed (4 vectors per reduction step,
// each reused across all R rows -> 24 accumulators at R=6).

// Scalar reference / fallback.
inline void qkt_tile_scalar(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int R,
    int Sk) {
  for (int r = 0; r < R; ++r) {
    const float* q = Q + r * q_row_stride;
    float* out = scores_buf + r * scores_row_stride;
    for (int j = 0; j < Sk; ++j) {
      float sum = 0.0f;
      for (int64_t e = 0; e < E; ++e) {
        sum += q[e] * Kp[e * 16 + j];
      }
      out[j] = sum;
    }
  }
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
template <int R>
inline void qkt_6x16_tile_neon(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int Sk) {
  static_assert(R >= 1 && R <= 6, "qkt_6x16_tile_neon supports 1..6 rows");
  float32x4_t acc[R][4];
  for (int r = 0; r < R; ++r) {
    acc[r][0] = vdupq_n_f32(0.0f);
    acc[r][1] = vdupq_n_f32(0.0f);
    acc[r][2] = vdupq_n_f32(0.0f);
    acc[r][3] = vdupq_n_f32(0.0f);
  }
  const float* qr[R];
  for (int r = 0; r < R; ++r) {
    qr[r] = Q + r * q_row_stride;
  }

  int64_t e = 0;
  // Reduction unroll-4: each Q vector holds 4 E-values, broadcast by lane.
#define FUSED_CPP_QKT6X16_KSTEP(KK)                                        \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 16;                                \
    const float32x4_t b0 = vld1q_f32(kb + 0);                              \
    const float32x4_t b1 = vld1q_f32(kb + 4);                              \
    const float32x4_t b2 = vld1q_f32(kb + 8);                              \
    const float32x4_t b3 = vld1q_f32(kb + 12);                             \
    for (int r = 0; r < R; ++r) {                                          \
      acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, qv[r], (KK));             \
      acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, qv[r], (KK));             \
      acc[r][2] = vfmaq_laneq_f32(acc[r][2], b2, qv[r], (KK));             \
      acc[r][3] = vfmaq_laneq_f32(acc[r][3], b3, qv[r], (KK));             \
    }                                                                      \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    float32x4_t qv[R];
    for (int r = 0; r < R; ++r) {
      qv[r] = vld1q_f32(qr[r] + e);
    }
    FUSED_CPP_QKT6X16_KSTEP(0);
    FUSED_CPP_QKT6X16_KSTEP(1);
    FUSED_CPP_QKT6X16_KSTEP(2);
    FUSED_CPP_QKT6X16_KSTEP(3);
  }
#undef FUSED_CPP_QKT6X16_KSTEP

  for (; e < E; ++e) {
    const float* kb = Kp + e * 16;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    const float32x4_t b2 = vld1q_f32(kb + 8);
    const float32x4_t b3 = vld1q_f32(kb + 12);
    for (int r = 0; r < R; ++r) {
      const float qf = qr[r][e];
      acc[r][0] = vfmaq_n_f32(acc[r][0], b0, qf);
      acc[r][1] = vfmaq_n_f32(acc[r][1], b1, qf);
      acc[r][2] = vfmaq_n_f32(acc[r][2], b2, qf);
      acc[r][3] = vfmaq_n_f32(acc[r][3], b3, qf);
    }
  }

  if (Sk == 16) {
    for (int r = 0; r < R; ++r) {
      float* out = scores_buf + r * scores_row_stride;
      vst1q_f32(out + 0, acc[r][0]);
      vst1q_f32(out + 4, acc[r][1]);
      vst1q_f32(out + 8, acc[r][2]);
      vst1q_f32(out + 12, acc[r][3]);
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float tmp[16];
      vst1q_f32(tmp + 0, acc[r][0]);
      vst1q_f32(tmp + 4, acc[r][1]);
      vst1q_f32(tmp + 8, acc[r][2]);
      vst1q_f32(tmp + 12, acc[r][3]);
      float* out = scores_buf + r * scores_row_stride;
      for (int j = 0; j < Sk; ++j) {
        out[j] = tmp[j];
      }
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
// Explicit 24-named-accumulator R=6 fast path. 24 score accumulators (a<r><c>),
// 6 held Q rows (qv0..qv5, 4 E-values each), B streamed via 2 regs (bA/bB).
// Mirrors the KleidiAI 6x16 schedule (v8-v31 / v6-v7) so clang keeps all 24
// accumulators in registers instead of spilling an array to the stack.
FUSED_CPP_PACK16_NOINLINE inline void qkt_6x16_neon_h6(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int Sk) {
  float32x4_t a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23,
      a30, a31, a32, a33, a40, a41, a42, a43, a50, a51, a52, a53;
  const float32x4_t z = vdupq_n_f32(0.0f);
  a00 = a01 = a02 = a03 = a10 = a11 = a12 = a13 = a20 = a21 = a22 = a23 =
      a30 = a31 = a32 = a33 = a40 = a41 = a42 = a43 = a50 = a51 = a52 = a53 = z;
  const float* q0 = Q;
  const float* q1 = Q + 1 * q_row_stride;
  const float* q2 = Q + 2 * q_row_stride;
  const float* q3 = Q + 3 * q_row_stride;
  const float* q4 = Q + 4 * q_row_stride;
  const float* q5 = Q + 5 * q_row_stride;

  int64_t e = 0;
  float32x4_t bA, bB;
#define FUSED_CPP_QKT6_COL(KK, COL, REG)                                   \
  do {                                                                     \
    REG = vld1q_f32(kb + (COL) * 4);                                       \
    a0##COL = vfmaq_laneq_f32(a0##COL, REG, qv0, (KK));                    \
    a1##COL = vfmaq_laneq_f32(a1##COL, REG, qv1, (KK));                    \
    a2##COL = vfmaq_laneq_f32(a2##COL, REG, qv2, (KK));                    \
    a3##COL = vfmaq_laneq_f32(a3##COL, REG, qv3, (KK));                    \
    a4##COL = vfmaq_laneq_f32(a4##COL, REG, qv4, (KK));                    \
    a5##COL = vfmaq_laneq_f32(a5##COL, REG, qv5, (KK));                    \
  } while (0)
#define FUSED_CPP_QKT6_KSTEP(KK)                                           \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 16;                                \
    FUSED_CPP_QKT6_COL(KK, 0, bA);                                         \
    FUSED_CPP_QKT6_COL(KK, 1, bB);                                         \
    FUSED_CPP_QKT6_COL(KK, 2, bA);                                         \
    FUSED_CPP_QKT6_COL(KK, 3, bB);                                         \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    const float32x4_t qv0 = vld1q_f32(q0 + e);
    const float32x4_t qv1 = vld1q_f32(q1 + e);
    const float32x4_t qv2 = vld1q_f32(q2 + e);
    const float32x4_t qv3 = vld1q_f32(q3 + e);
    const float32x4_t qv4 = vld1q_f32(q4 + e);
    const float32x4_t qv5 = vld1q_f32(q5 + e);
    FUSED_CPP_QKT6_KSTEP(0);
    FUSED_CPP_QKT6_KSTEP(1);
    FUSED_CPP_QKT6_KSTEP(2);
    FUSED_CPP_QKT6_KSTEP(3);
  }
#undef FUSED_CPP_QKT6_KSTEP
#undef FUSED_CPP_QKT6_COL

  for (; e < E; ++e) {
    const float* kb = Kp + e * 16;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    const float32x4_t b2 = vld1q_f32(kb + 8);
    const float32x4_t b3 = vld1q_f32(kb + 12);
    const float f0 = q0[e], f1 = q1[e], f2 = q2[e], f3 = q3[e], f4 = q4[e], f5 = q5[e];
    a00 = vfmaq_n_f32(a00, b0, f0); a01 = vfmaq_n_f32(a01, b1, f0); a02 = vfmaq_n_f32(a02, b2, f0); a03 = vfmaq_n_f32(a03, b3, f0);
    a10 = vfmaq_n_f32(a10, b0, f1); a11 = vfmaq_n_f32(a11, b1, f1); a12 = vfmaq_n_f32(a12, b2, f1); a13 = vfmaq_n_f32(a13, b3, f1);
    a20 = vfmaq_n_f32(a20, b0, f2); a21 = vfmaq_n_f32(a21, b1, f2); a22 = vfmaq_n_f32(a22, b2, f2); a23 = vfmaq_n_f32(a23, b3, f2);
    a30 = vfmaq_n_f32(a30, b0, f3); a31 = vfmaq_n_f32(a31, b1, f3); a32 = vfmaq_n_f32(a32, b2, f3); a33 = vfmaq_n_f32(a33, b3, f3);
    a40 = vfmaq_n_f32(a40, b0, f4); a41 = vfmaq_n_f32(a41, b1, f4); a42 = vfmaq_n_f32(a42, b2, f4); a43 = vfmaq_n_f32(a43, b3, f4);
    a50 = vfmaq_n_f32(a50, b0, f5); a51 = vfmaq_n_f32(a51, b1, f5); a52 = vfmaq_n_f32(a52, b2, f5); a53 = vfmaq_n_f32(a53, b3, f5);
  }

  float32x4_t rows[6][4] = {
      {a00, a01, a02, a03}, {a10, a11, a12, a13}, {a20, a21, a22, a23},
      {a30, a31, a32, a33}, {a40, a41, a42, a43}, {a50, a51, a52, a53}};
  if (Sk == 16) {
    for (int r = 0; r < 6; ++r) {
      float* out = scores_buf + r * scores_row_stride;
      vst1q_f32(out + 0, rows[r][0]);
      vst1q_f32(out + 4, rows[r][1]);
      vst1q_f32(out + 8, rows[r][2]);
      vst1q_f32(out + 12, rows[r][3]);
    }
  } else {
    for (int r = 0; r < 6; ++r) {
      float tmp[16];
      vst1q_f32(tmp + 0, rows[r][0]);
      vst1q_f32(tmp + 4, rows[r][1]);
      vst1q_f32(tmp + 8, rows[r][2]);
      vst1q_f32(tmp + 12, rows[r][3]);
      float* out = scores_buf + r * scores_row_stride;
      for (int j = 0; j < Sk; ++j) out[j] = tmp[j];
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// Dispatch one tile: R rows (1..6) x Sk cols (1..16).
inline void qkt_6x16_tile(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int R,
    int Sk) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  switch (R) {
    case 6: qkt_6x16_neon_h6(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 5: qkt_6x16_tile_neon<5>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 4: qkt_6x16_tile_neon<4>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 3: qkt_6x16_tile_neon<3>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 2: qkt_6x16_tile_neon<2>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 1: qkt_6x16_tile_neon<1>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    default: break;
  }
#endif
  qkt_tile_scalar(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, R, Sk);
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
// Row-max-fused QKT for a FULL 16-col S-block. Computes scores[R x 16] and folds
// the per-row running max into row_max[r] (kept in vector form; the caller
// reduces each to a scalar after the last S-block). Partial (Sk<16) blocks use
// the plain qkt_6x16_tile + a masked max in the orchestration.
template <int R>
inline void qkt_6x16_rowmax_tile_neon(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    float32x4_t row_max[6]) {
  static_assert(R >= 1 && R <= 6, "qkt_6x16_rowmax_tile_neon supports 1..6 rows");
  float32x4_t acc[R][4];
  for (int r = 0; r < R; ++r) {
    acc[r][0] = vdupq_n_f32(0.0f);
    acc[r][1] = vdupq_n_f32(0.0f);
    acc[r][2] = vdupq_n_f32(0.0f);
    acc[r][3] = vdupq_n_f32(0.0f);
  }
  const float* qr[R];
  for (int r = 0; r < R; ++r) {
    qr[r] = Q + r * q_row_stride;
  }

  int64_t e = 0;
#define FUSED_CPP_QKT6X16RM_KSTEP(KK)                                      \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 16;                                \
    const float32x4_t b0 = vld1q_f32(kb + 0);                              \
    const float32x4_t b1 = vld1q_f32(kb + 4);                              \
    const float32x4_t b2 = vld1q_f32(kb + 8);                              \
    const float32x4_t b3 = vld1q_f32(kb + 12);                             \
    for (int r = 0; r < R; ++r) {                                          \
      acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, qv[r], (KK));             \
      acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, qv[r], (KK));             \
      acc[r][2] = vfmaq_laneq_f32(acc[r][2], b2, qv[r], (KK));             \
      acc[r][3] = vfmaq_laneq_f32(acc[r][3], b3, qv[r], (KK));             \
    }                                                                      \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    float32x4_t qv[R];
    for (int r = 0; r < R; ++r) {
      qv[r] = vld1q_f32(qr[r] + e);
    }
    FUSED_CPP_QKT6X16RM_KSTEP(0);
    FUSED_CPP_QKT6X16RM_KSTEP(1);
    FUSED_CPP_QKT6X16RM_KSTEP(2);
    FUSED_CPP_QKT6X16RM_KSTEP(3);
  }
#undef FUSED_CPP_QKT6X16RM_KSTEP

  for (; e < E; ++e) {
    const float* kb = Kp + e * 16;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    const float32x4_t b2 = vld1q_f32(kb + 8);
    const float32x4_t b3 = vld1q_f32(kb + 12);
    for (int r = 0; r < R; ++r) {
      const float qf = qr[r][e];
      acc[r][0] = vfmaq_n_f32(acc[r][0], b0, qf);
      acc[r][1] = vfmaq_n_f32(acc[r][1], b1, qf);
      acc[r][2] = vfmaq_n_f32(acc[r][2], b2, qf);
      acc[r][3] = vfmaq_n_f32(acc[r][3], b3, qf);
    }
  }

  for (int r = 0; r < R; ++r) {
    float* out = scores_buf + r * scores_row_stride;
    vst1q_f32(out + 0, acc[r][0]);
    vst1q_f32(out + 4, acc[r][1]);
    vst1q_f32(out + 8, acc[r][2]);
    vst1q_f32(out + 12, acc[r][3]);
    const float32x4_t m01 = vmaxq_f32(acc[r][0], acc[r][1]);
    const float32x4_t m23 = vmaxq_f32(acc[r][2], acc[r][3]);
    row_max[r] = vmaxq_f32(row_max[r], vmaxq_f32(m01, m23));
  }
}

// Explicit 24-named-accumulator R=6 row-max-fused QKT (full 16-col block).
FUSED_CPP_PACK16_NOINLINE inline void qkt_6x16_rowmax_neon_h6(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    float32x4_t row_max[6]) {
  float32x4_t a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23,
      a30, a31, a32, a33, a40, a41, a42, a43, a50, a51, a52, a53;
  const float32x4_t z = vdupq_n_f32(0.0f);
  a00 = a01 = a02 = a03 = a10 = a11 = a12 = a13 = a20 = a21 = a22 = a23 =
      a30 = a31 = a32 = a33 = a40 = a41 = a42 = a43 = a50 = a51 = a52 = a53 = z;
  const float* q0 = Q; const float* q1 = Q + 1 * q_row_stride;
  const float* q2 = Q + 2 * q_row_stride; const float* q3 = Q + 3 * q_row_stride;
  const float* q4 = Q + 4 * q_row_stride; const float* q5 = Q + 5 * q_row_stride;

  int64_t e = 0;
  float32x4_t bA, bB;
#define FUSED_CPP_QKT6RM_COL(KK, COL, REG)                                 \
  do {                                                                     \
    REG = vld1q_f32(kb + (COL) * 4);                                       \
    a0##COL = vfmaq_laneq_f32(a0##COL, REG, qv0, (KK));                    \
    a1##COL = vfmaq_laneq_f32(a1##COL, REG, qv1, (KK));                    \
    a2##COL = vfmaq_laneq_f32(a2##COL, REG, qv2, (KK));                    \
    a3##COL = vfmaq_laneq_f32(a3##COL, REG, qv3, (KK));                    \
    a4##COL = vfmaq_laneq_f32(a4##COL, REG, qv4, (KK));                    \
    a5##COL = vfmaq_laneq_f32(a5##COL, REG, qv5, (KK));                    \
  } while (0)
#define FUSED_CPP_QKT6RM_KSTEP(KK)                                         \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 16;                                \
    FUSED_CPP_QKT6RM_COL(KK, 0, bA);                                       \
    FUSED_CPP_QKT6RM_COL(KK, 1, bB);                                       \
    FUSED_CPP_QKT6RM_COL(KK, 2, bA);                                       \
    FUSED_CPP_QKT6RM_COL(KK, 3, bB);                                       \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    const float32x4_t qv0 = vld1q_f32(q0 + e);
    const float32x4_t qv1 = vld1q_f32(q1 + e);
    const float32x4_t qv2 = vld1q_f32(q2 + e);
    const float32x4_t qv3 = vld1q_f32(q3 + e);
    const float32x4_t qv4 = vld1q_f32(q4 + e);
    const float32x4_t qv5 = vld1q_f32(q5 + e);
    FUSED_CPP_QKT6RM_KSTEP(0);
    FUSED_CPP_QKT6RM_KSTEP(1);
    FUSED_CPP_QKT6RM_KSTEP(2);
    FUSED_CPP_QKT6RM_KSTEP(3);
  }
#undef FUSED_CPP_QKT6RM_KSTEP
#undef FUSED_CPP_QKT6RM_COL

  for (; e < E; ++e) {
    const float* kb = Kp + e * 16;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    const float32x4_t b2 = vld1q_f32(kb + 8);
    const float32x4_t b3 = vld1q_f32(kb + 12);
    const float f0 = q0[e], f1 = q1[e], f2 = q2[e], f3 = q3[e], f4 = q4[e], f5 = q5[e];
    a00 = vfmaq_n_f32(a00, b0, f0); a01 = vfmaq_n_f32(a01, b1, f0); a02 = vfmaq_n_f32(a02, b2, f0); a03 = vfmaq_n_f32(a03, b3, f0);
    a10 = vfmaq_n_f32(a10, b0, f1); a11 = vfmaq_n_f32(a11, b1, f1); a12 = vfmaq_n_f32(a12, b2, f1); a13 = vfmaq_n_f32(a13, b3, f1);
    a20 = vfmaq_n_f32(a20, b0, f2); a21 = vfmaq_n_f32(a21, b1, f2); a22 = vfmaq_n_f32(a22, b2, f2); a23 = vfmaq_n_f32(a23, b3, f2);
    a30 = vfmaq_n_f32(a30, b0, f3); a31 = vfmaq_n_f32(a31, b1, f3); a32 = vfmaq_n_f32(a32, b2, f3); a33 = vfmaq_n_f32(a33, b3, f3);
    a40 = vfmaq_n_f32(a40, b0, f4); a41 = vfmaq_n_f32(a41, b1, f4); a42 = vfmaq_n_f32(a42, b2, f4); a43 = vfmaq_n_f32(a43, b3, f4);
    a50 = vfmaq_n_f32(a50, b0, f5); a51 = vfmaq_n_f32(a51, b1, f5); a52 = vfmaq_n_f32(a52, b2, f5); a53 = vfmaq_n_f32(a53, b3, f5);
  }

  float32x4_t rows[6][4] = {
      {a00, a01, a02, a03}, {a10, a11, a12, a13}, {a20, a21, a22, a23},
      {a30, a31, a32, a33}, {a40, a41, a42, a43}, {a50, a51, a52, a53}};
  for (int r = 0; r < 6; ++r) {
    float* out = scores_buf + r * scores_row_stride;
    vst1q_f32(out + 0, rows[r][0]);
    vst1q_f32(out + 4, rows[r][1]);
    vst1q_f32(out + 8, rows[r][2]);
    vst1q_f32(out + 12, rows[r][3]);
    const float32x4_t m01 = vmaxq_f32(rows[r][0], rows[r][1]);
    const float32x4_t m23 = vmaxq_f32(rows[r][2], rows[r][3]);
    row_max[r] = vmaxq_f32(row_max[r], vmaxq_f32(m01, m23));
  }
}

// Vector-interface row-max-fused dispatch: folds into row_max[6] (kept across
// S-blocks by the caller; reduced to scalar once after the last block). Avoids
// the per-call vdup/vmaxvq round-trip of the scalar interface below.
inline void qkt_6x16_rowmax_vec(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int R,
    float32x4_t row_max[6]) {
  switch (R) {
    case 6: qkt_6x16_rowmax_neon_h6(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 5: qkt_6x16_rowmax_tile_neon<5>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 4: qkt_6x16_rowmax_tile_neon<4>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 3: qkt_6x16_rowmax_tile_neon<3>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 2: qkt_6x16_rowmax_tile_neon<2>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 1: qkt_6x16_rowmax_tile_neon<1>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    default: return;
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// Dispatch a full-16 row-max-fused tile (R rows, Sk==16).
inline void qkt_6x16_rowmax(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int R,
    float* row_max_scalar) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  float32x4_t rm[6];
  for (int r = 0; r < R; ++r) rm[r] = vdupq_n_f32(row_max_scalar[r]);
  switch (R) {
    case 6: qkt_6x16_rowmax_neon_h6(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, rm); break;
    case 5: qkt_6x16_rowmax_tile_neon<5>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, rm); break;
    case 4: qkt_6x16_rowmax_tile_neon<4>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, rm); break;
    case 3: qkt_6x16_rowmax_tile_neon<3>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, rm); break;
    case 2: qkt_6x16_rowmax_tile_neon<2>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, rm); break;
    case 1: qkt_6x16_rowmax_tile_neon<1>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, rm); break;
    default: break;
  }
  for (int r = 0; r < R; ++r) row_max_scalar[r] = vmaxvq_f32(rm[r]);
  return;
#else
  qkt_tile_scalar(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, R, 16);
  for (int r = 0; r < R; ++r) {
    float m = row_max_scalar[r];
    const float* out = scores_buf + r * scores_row_stride;
    for (int j = 0; j < 16; ++j) m = out[j] > m ? out[j] : m;
    row_max_scalar[r] = m;
  }
#endif
}

// ── PV 6x16 stream-B kernel ────────────────────────────────────────────────
//
// O[R x Ev_cur] += P[R x Sk] (P_row_stride) . Vp, where Vp is one Ev-block of
// packed V ([S][16], s-major). Reduction over the S-tile (k=0..Sk). P rows are
// held (lane-broadcast, unroll-4 over S); the 16 packed V cols are streamed.
// ACCUMULATES into O (flash2 rescales O before each call / multiple segments).

inline void pv_tile_scalar_acc(
    const float* P,
    int64_t P_row_stride,
    const float* Vp,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
    int64_t o_row_stride,
    int R,
    int Ev_cur) {
  for (int r = 0; r < R; ++r) {
    const float* p = P + r * P_row_stride;
    float* o = O + r * o_row_stride;
    for (int j = 0; j < Ev_cur; ++j) {
      float sum = 0.0f;
      for (int64_t s = 0; s < Sk; ++s) {
        sum += p[s] * Vp[s * v_row_stride + j];
      }
      o[j] += sum;
    }
  }
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
template <int R>
inline void pv_6x16_tile_neon(
    const float* P,
    int64_t P_row_stride,
    const float* Vp,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
    int64_t o_row_stride,
    int Ev_cur) {
  static_assert(R >= 1 && R <= 6, "pv_6x16_tile_neon supports 1..6 rows");
  float32x4_t acc[R][4];
  // Seed accumulators from existing O (accumulate semantics).
  if (Ev_cur == 16) {
    for (int r = 0; r < R; ++r) {
      const float* o = O + r * o_row_stride;
      acc[r][0] = vld1q_f32(o + 0);
      acc[r][1] = vld1q_f32(o + 4);
      acc[r][2] = vld1q_f32(o + 8);
      acc[r][3] = vld1q_f32(o + 12);
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float tmp[16] = {0};
      const float* o = O + r * o_row_stride;
      for (int j = 0; j < Ev_cur; ++j) tmp[j] = o[j];
      acc[r][0] = vld1q_f32(tmp + 0);
      acc[r][1] = vld1q_f32(tmp + 4);
      acc[r][2] = vld1q_f32(tmp + 8);
      acc[r][3] = vld1q_f32(tmp + 12);
    }
  }

  const float* pr[R];
  for (int r = 0; r < R; ++r) {
    pr[r] = P + r * P_row_stride;
  }

  int64_t s = 0;
#define FUSED_CPP_PV6X16_VSTEP(KK)                                         \
  do {                                                                     \
    const float* vb = Vp + (s + (KK)) * v_row_stride;                      \
    const float32x4_t b0 = vld1q_f32(vb + 0);                              \
    const float32x4_t b1 = vld1q_f32(vb + 4);                              \
    const float32x4_t b2 = vld1q_f32(vb + 8);                              \
    const float32x4_t b3 = vld1q_f32(vb + 12);                             \
    for (int r = 0; r < R; ++r) {                                          \
      acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, pvv[r], (KK));            \
      acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, pvv[r], (KK));            \
      acc[r][2] = vfmaq_laneq_f32(acc[r][2], b2, pvv[r], (KK));            \
      acc[r][3] = vfmaq_laneq_f32(acc[r][3], b3, pvv[r], (KK));            \
    }                                                                      \
  } while (0)

  for (; s + 4 <= Sk; s += 4) {
    float32x4_t pvv[R];
    for (int r = 0; r < R; ++r) {
      pvv[r] = vld1q_f32(pr[r] + s);
    }
    FUSED_CPP_PV6X16_VSTEP(0);
    FUSED_CPP_PV6X16_VSTEP(1);
    FUSED_CPP_PV6X16_VSTEP(2);
    FUSED_CPP_PV6X16_VSTEP(3);
  }
#undef FUSED_CPP_PV6X16_VSTEP

  for (; s < Sk; ++s) {
    const float* vb = Vp + s * v_row_stride;
    const float32x4_t b0 = vld1q_f32(vb + 0);
    const float32x4_t b1 = vld1q_f32(vb + 4);
    const float32x4_t b2 = vld1q_f32(vb + 8);
    const float32x4_t b3 = vld1q_f32(vb + 12);
    for (int r = 0; r < R; ++r) {
      const float pf = pr[r][s];
      acc[r][0] = vfmaq_n_f32(acc[r][0], b0, pf);
      acc[r][1] = vfmaq_n_f32(acc[r][1], b1, pf);
      acc[r][2] = vfmaq_n_f32(acc[r][2], b2, pf);
      acc[r][3] = vfmaq_n_f32(acc[r][3], b3, pf);
    }
  }

  if (Ev_cur == 16) {
    for (int r = 0; r < R; ++r) {
      float* o = O + r * o_row_stride;
      vst1q_f32(o + 0, acc[r][0]);
      vst1q_f32(o + 4, acc[r][1]);
      vst1q_f32(o + 8, acc[r][2]);
      vst1q_f32(o + 12, acc[r][3]);
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float tmp[16];
      vst1q_f32(tmp + 0, acc[r][0]);
      vst1q_f32(tmp + 4, acc[r][1]);
      vst1q_f32(tmp + 8, acc[r][2]);
      vst1q_f32(tmp + 12, acc[r][3]);
      float* o = O + r * o_row_stride;
      for (int j = 0; j < Ev_cur; ++j) o[j] = tmp[j];
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
// Explicit 24-named-accumulator R=6 PV fast path (accumulate into O). 6 held P
// rows (pv0..pv5, 4 S-values each), V streamed via 2 regs (bA/bB).
FUSED_CPP_PACK16_NOINLINE inline void pv_6x16_neon_h6(
    const float* P,
    int64_t P_row_stride,
    const float* Vp,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
    int64_t o_row_stride,
    int Ev_cur) {
  float32x4_t a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23,
      a30, a31, a32, a33, a40, a41, a42, a43, a50, a51, a52, a53;
  if (Ev_cur == 16) {
    const float* o0 = O; const float* o1 = O + o_row_stride;
    const float* o2 = O + 2 * o_row_stride; const float* o3 = O + 3 * o_row_stride;
    const float* o4 = O + 4 * o_row_stride; const float* o5 = O + 5 * o_row_stride;
    a00 = vld1q_f32(o0 + 0); a01 = vld1q_f32(o0 + 4); a02 = vld1q_f32(o0 + 8); a03 = vld1q_f32(o0 + 12);
    a10 = vld1q_f32(o1 + 0); a11 = vld1q_f32(o1 + 4); a12 = vld1q_f32(o1 + 8); a13 = vld1q_f32(o1 + 12);
    a20 = vld1q_f32(o2 + 0); a21 = vld1q_f32(o2 + 4); a22 = vld1q_f32(o2 + 8); a23 = vld1q_f32(o2 + 12);
    a30 = vld1q_f32(o3 + 0); a31 = vld1q_f32(o3 + 4); a32 = vld1q_f32(o3 + 8); a33 = vld1q_f32(o3 + 12);
    a40 = vld1q_f32(o4 + 0); a41 = vld1q_f32(o4 + 4); a42 = vld1q_f32(o4 + 8); a43 = vld1q_f32(o4 + 12);
    a50 = vld1q_f32(o5 + 0); a51 = vld1q_f32(o5 + 4); a52 = vld1q_f32(o5 + 8); a53 = vld1q_f32(o5 + 12);
  } else {
    float tmp[6][16] = {{0}};
    for (int r = 0; r < 6; ++r)
      for (int j = 0; j < Ev_cur; ++j) tmp[r][j] = O[r * o_row_stride + j];
    a00 = vld1q_f32(tmp[0] + 0); a01 = vld1q_f32(tmp[0] + 4); a02 = vld1q_f32(tmp[0] + 8); a03 = vld1q_f32(tmp[0] + 12);
    a10 = vld1q_f32(tmp[1] + 0); a11 = vld1q_f32(tmp[1] + 4); a12 = vld1q_f32(tmp[1] + 8); a13 = vld1q_f32(tmp[1] + 12);
    a20 = vld1q_f32(tmp[2] + 0); a21 = vld1q_f32(tmp[2] + 4); a22 = vld1q_f32(tmp[2] + 8); a23 = vld1q_f32(tmp[2] + 12);
    a30 = vld1q_f32(tmp[3] + 0); a31 = vld1q_f32(tmp[3] + 4); a32 = vld1q_f32(tmp[3] + 8); a33 = vld1q_f32(tmp[3] + 12);
    a40 = vld1q_f32(tmp[4] + 0); a41 = vld1q_f32(tmp[4] + 4); a42 = vld1q_f32(tmp[4] + 8); a43 = vld1q_f32(tmp[4] + 12);
    a50 = vld1q_f32(tmp[5] + 0); a51 = vld1q_f32(tmp[5] + 4); a52 = vld1q_f32(tmp[5] + 8); a53 = vld1q_f32(tmp[5] + 12);
  }
  const float* p0 = P; const float* p1 = P + 1 * P_row_stride;
  const float* p2 = P + 2 * P_row_stride; const float* p3 = P + 3 * P_row_stride;
  const float* p4 = P + 4 * P_row_stride; const float* p5 = P + 5 * P_row_stride;

  int64_t s = 0;
  float32x4_t bA, bB;
#define FUSED_CPP_PV6_COL(KK, COL, REG)                                    \
  do {                                                                     \
    REG = vld1q_f32(vb + (COL) * 4);                                       \
    a0##COL = vfmaq_laneq_f32(a0##COL, REG, pv0, (KK));                    \
    a1##COL = vfmaq_laneq_f32(a1##COL, REG, pv1, (KK));                    \
    a2##COL = vfmaq_laneq_f32(a2##COL, REG, pv2, (KK));                    \
    a3##COL = vfmaq_laneq_f32(a3##COL, REG, pv3, (KK));                    \
    a4##COL = vfmaq_laneq_f32(a4##COL, REG, pv4, (KK));                    \
    a5##COL = vfmaq_laneq_f32(a5##COL, REG, pv5, (KK));                    \
  } while (0)
#define FUSED_CPP_PV6_SSTEP(KK)                                            \
  do {                                                                     \
    const float* vb = Vp + (s + (KK)) * v_row_stride;                      \
    FUSED_CPP_PV6_COL(KK, 0, bA);                                          \
    FUSED_CPP_PV6_COL(KK, 1, bB);                                          \
    FUSED_CPP_PV6_COL(KK, 2, bA);                                          \
    FUSED_CPP_PV6_COL(KK, 3, bB);                                          \
  } while (0)

  for (; s + 4 <= Sk; s += 4) {
    const float32x4_t pv0 = vld1q_f32(p0 + s);
    const float32x4_t pv1 = vld1q_f32(p1 + s);
    const float32x4_t pv2 = vld1q_f32(p2 + s);
    const float32x4_t pv3 = vld1q_f32(p3 + s);
    const float32x4_t pv4 = vld1q_f32(p4 + s);
    const float32x4_t pv5 = vld1q_f32(p5 + s);
    FUSED_CPP_PV6_SSTEP(0);
    FUSED_CPP_PV6_SSTEP(1);
    FUSED_CPP_PV6_SSTEP(2);
    FUSED_CPP_PV6_SSTEP(3);
  }
#undef FUSED_CPP_PV6_SSTEP
#undef FUSED_CPP_PV6_COL

  for (; s < Sk; ++s) {
    const float* vb = Vp + s * v_row_stride;
    const float32x4_t b0 = vld1q_f32(vb + 0);
    const float32x4_t b1 = vld1q_f32(vb + 4);
    const float32x4_t b2 = vld1q_f32(vb + 8);
    const float32x4_t b3 = vld1q_f32(vb + 12);
    const float f0 = p0[s], f1 = p1[s], f2 = p2[s], f3 = p3[s], f4 = p4[s], f5 = p5[s];
    a00 = vfmaq_n_f32(a00, b0, f0); a01 = vfmaq_n_f32(a01, b1, f0); a02 = vfmaq_n_f32(a02, b2, f0); a03 = vfmaq_n_f32(a03, b3, f0);
    a10 = vfmaq_n_f32(a10, b0, f1); a11 = vfmaq_n_f32(a11, b1, f1); a12 = vfmaq_n_f32(a12, b2, f1); a13 = vfmaq_n_f32(a13, b3, f1);
    a20 = vfmaq_n_f32(a20, b0, f2); a21 = vfmaq_n_f32(a21, b1, f2); a22 = vfmaq_n_f32(a22, b2, f2); a23 = vfmaq_n_f32(a23, b3, f2);
    a30 = vfmaq_n_f32(a30, b0, f3); a31 = vfmaq_n_f32(a31, b1, f3); a32 = vfmaq_n_f32(a32, b2, f3); a33 = vfmaq_n_f32(a33, b3, f3);
    a40 = vfmaq_n_f32(a40, b0, f4); a41 = vfmaq_n_f32(a41, b1, f4); a42 = vfmaq_n_f32(a42, b2, f4); a43 = vfmaq_n_f32(a43, b3, f4);
    a50 = vfmaq_n_f32(a50, b0, f5); a51 = vfmaq_n_f32(a51, b1, f5); a52 = vfmaq_n_f32(a52, b2, f5); a53 = vfmaq_n_f32(a53, b3, f5);
  }

  float32x4_t rows[6][4] = {
      {a00, a01, a02, a03}, {a10, a11, a12, a13}, {a20, a21, a22, a23},
      {a30, a31, a32, a33}, {a40, a41, a42, a43}, {a50, a51, a52, a53}};
  if (Ev_cur == 16) {
    for (int r = 0; r < 6; ++r) {
      float* o = O + r * o_row_stride;
      vst1q_f32(o + 0, rows[r][0]);
      vst1q_f32(o + 4, rows[r][1]);
      vst1q_f32(o + 8, rows[r][2]);
      vst1q_f32(o + 12, rows[r][3]);
    }
  } else {
    for (int r = 0; r < 6; ++r) {
      float tmp[16];
      vst1q_f32(tmp + 0, rows[r][0]);
      vst1q_f32(tmp + 4, rows[r][1]);
      vst1q_f32(tmp + 8, rows[r][2]);
      vst1q_f32(tmp + 12, rows[r][3]);
      float* o = O + r * o_row_stride;
      for (int j = 0; j < Ev_cur; ++j) o[j] = tmp[j];
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// Dispatch one PV tile: R rows (1..6) x Ev_cur cols (1..16), reduction Sk.
inline void pv_6x16_tile(
    const float* P,
    int64_t P_row_stride,
    const float* Vp,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
    int64_t o_row_stride,
    int R,
    int Ev_cur) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  switch (R) {
    case 6: pv_6x16_neon_h6(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 5: pv_6x16_tile_neon<5>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 4: pv_6x16_tile_neon<4>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 3: pv_6x16_tile_neon<3>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 2: pv_6x16_tile_neon<2>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 1: pv_6x16_tile_neon<1>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    default: break;
  }
#endif
  pv_tile_scalar_acc(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, R, Ev_cur);
}

}  // namespace sdpa_pack16
}  // namespace fused_cpp
