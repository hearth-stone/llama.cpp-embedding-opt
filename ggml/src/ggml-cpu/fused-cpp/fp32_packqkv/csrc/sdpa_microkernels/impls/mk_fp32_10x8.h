#pragma once
// 10x8 stream-B fp32 microkernels for the fp32 packqkv SDPA path. Reuses the
// existing width-8 packers (sblock8 K, evblock8 V); this header holds KERNELS
// ONLY. Header-only so it can be unit-tested standalone and reached from the
// single fp32_packqkv_sdpa.cpp translation unit.
//
// Packed layouts (per head, produced by the existing width-8 packers):
//   K (QKT RHS): [S_b8][E][8]  dst[sb*E*8 + e*8 + j] = scale * K[sb*8+j, e]
//                (e-major within an S-block of 8 keys; scale folded in)
//   V (PV  RHS): [Ev_b8][S][8] dst[evb*S*8 + s*8 + j] = V[s, evb*8+j]
//                (s-major within an Ev-block of 8; reduction dim S is contiguous)
//
// Both feed a 10x8 stream-B kernel that holds 10 LHS rows + 20 accumulators and
// streams the packed RHS through 2 NEON registers. Live regs = 20 + 10 + 2 = 32
// (exact fit) -> explicit named accumulators + NOINLINE are mandatory or clang
// spills the accumulator array.

#include <cstdint>

#include "sdpa_microkernels/neon_cache_config.h"

namespace fused_cpp {
namespace sdpa_10x8 {

#if defined(__GNUC__) || defined(__clang__)
#define FUSED_CPP_PACK8_NOINLINE __attribute__((noinline))
#else
#define FUSED_CPP_PACK8_NOINLINE
#endif

constexpr int kMR = 10;  // rows of LHS held per tile
constexpr int kNR = 8;   // cols of RHS produced per tile

inline int64_t ceil_div8(int64_t v) { return (v + 7) / 8; }

inline int64_t k_packed_size_per_head(int64_t S, int64_t E) {
  return ceil_div8(S) * E * 8;
}

inline int64_t v_packed_size_per_head(int64_t S, int64_t Ev) {
  return ceil_div8(Ev) * S * 8;
}

// ── QKT 10x8 stream-B kernel ────────────────────────────────────────────────
//
// scores[R x Sk] = Q[R x E] (q_row_stride) . Kp, where Kp is one S-block of the
// packed K ([E][8], e-major). Scale is already folded into Kp. R = active rows
// (1..10), Sk = active cols (1..8). Q rows are held (lane-broadcast, unroll-4
// over E); the 8 packed K cols are streamed through 2 vectors (bA/bB).

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
        sum += q[e] * Kp[e * 8 + j];
      }
      out[j] = sum;
    }
  }
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
template <int R>
inline void qkt_10x8_tile_neon(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int Sk) {
  static_assert(R >= 1 && R <= 10, "qkt_10x8_tile_neon supports 1..10 rows");
  float32x4_t acc[R][2];
  for (int r = 0; r < R; ++r) {
    acc[r][0] = vdupq_n_f32(0.0f);
    acc[r][1] = vdupq_n_f32(0.0f);
  }
  const float* qr[R];
  for (int r = 0; r < R; ++r) {
    qr[r] = Q + r * q_row_stride;
  }

  int64_t e = 0;
#define FUSED_CPP_QKT10X8_KSTEP(KK)                                        \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 8;                                 \
    const float32x4_t b0 = vld1q_f32(kb + 0);                              \
    const float32x4_t b1 = vld1q_f32(kb + 4);                              \
    for (int r = 0; r < R; ++r) {                                          \
      acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, qv[r], (KK));             \
      acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, qv[r], (KK));             \
    }                                                                      \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    float32x4_t qv[R];
    for (int r = 0; r < R; ++r) {
      qv[r] = vld1q_f32(qr[r] + e);
    }
    FUSED_CPP_QKT10X8_KSTEP(0);
    FUSED_CPP_QKT10X8_KSTEP(1);
    FUSED_CPP_QKT10X8_KSTEP(2);
    FUSED_CPP_QKT10X8_KSTEP(3);
  }
#undef FUSED_CPP_QKT10X8_KSTEP

  for (; e < E; ++e) {
    const float* kb = Kp + e * 8;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    for (int r = 0; r < R; ++r) {
      const float qf = qr[r][e];
      acc[r][0] = vfmaq_n_f32(acc[r][0], b0, qf);
      acc[r][1] = vfmaq_n_f32(acc[r][1], b1, qf);
    }
  }

  if (Sk == 8) {
    for (int r = 0; r < R; ++r) {
      float* out = scores_buf + r * scores_row_stride;
      vst1q_f32(out + 0, acc[r][0]);
      vst1q_f32(out + 4, acc[r][1]);
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float tmp[8];
      vst1q_f32(tmp + 0, acc[r][0]);
      vst1q_f32(tmp + 4, acc[r][1]);
      float* out = scores_buf + r * scores_row_stride;
      for (int j = 0; j < Sk; ++j) out[j] = tmp[j];
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
// Explicit 20-named-accumulator R=10 fast path. 20 score accumulators (aRcC),
// 10 held Q rows (qv0..qv9, 4 E-values each), B streamed via 2 regs (bA/bB).
// Keeps all 20 accumulators in registers instead of spilling an array.
FUSED_CPP_PACK8_NOINLINE inline void qkt_10x8_neon_h10(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int Sk) {
  float32x4_t a0c0, a0c1, a1c0, a1c1, a2c0, a2c1, a3c0, a3c1, a4c0, a4c1,
      a5c0, a5c1, a6c0, a6c1, a7c0, a7c1, a8c0, a8c1, a9c0, a9c1;
  const float32x4_t z = vdupq_n_f32(0.0f);
  a0c0 = a0c1 = a1c0 = a1c1 = a2c0 = a2c1 = a3c0 = a3c1 = a4c0 = a4c1 =
      a5c0 = a5c1 = a6c0 = a6c1 = a7c0 = a7c1 = a8c0 = a8c1 = a9c0 = a9c1 = z;
  const float* q0 = Q;
  const float* q1 = Q + 1 * q_row_stride;
  const float* q2 = Q + 2 * q_row_stride;
  const float* q3 = Q + 3 * q_row_stride;
  const float* q4 = Q + 4 * q_row_stride;
  const float* q5 = Q + 5 * q_row_stride;
  const float* q6 = Q + 6 * q_row_stride;
  const float* q7 = Q + 7 * q_row_stride;
  const float* q8 = Q + 8 * q_row_stride;
  const float* q9 = Q + 9 * q_row_stride;

  int64_t e = 0;
  float32x4_t bA, bB;
#define FUSED_CPP_QKT10_COL(KK, COL, REG)                                  \
  do {                                                                     \
    REG = vld1q_f32(kb + (COL) * 4);                                       \
    a0c##COL = vfmaq_laneq_f32(a0c##COL, REG, qv0, (KK));                  \
    a1c##COL = vfmaq_laneq_f32(a1c##COL, REG, qv1, (KK));                  \
    a2c##COL = vfmaq_laneq_f32(a2c##COL, REG, qv2, (KK));                  \
    a3c##COL = vfmaq_laneq_f32(a3c##COL, REG, qv3, (KK));                  \
    a4c##COL = vfmaq_laneq_f32(a4c##COL, REG, qv4, (KK));                  \
    a5c##COL = vfmaq_laneq_f32(a5c##COL, REG, qv5, (KK));                  \
    a6c##COL = vfmaq_laneq_f32(a6c##COL, REG, qv6, (KK));                  \
    a7c##COL = vfmaq_laneq_f32(a7c##COL, REG, qv7, (KK));                  \
    a8c##COL = vfmaq_laneq_f32(a8c##COL, REG, qv8, (KK));                  \
    a9c##COL = vfmaq_laneq_f32(a9c##COL, REG, qv9, (KK));                  \
  } while (0)
#define FUSED_CPP_QKT10_KSTEP(KK)                                          \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 8;                                 \
    FUSED_CPP_QKT10_COL(KK, 0, bA);                                        \
    FUSED_CPP_QKT10_COL(KK, 1, bB);                                        \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    const float32x4_t qv0 = vld1q_f32(q0 + e);
    const float32x4_t qv1 = vld1q_f32(q1 + e);
    const float32x4_t qv2 = vld1q_f32(q2 + e);
    const float32x4_t qv3 = vld1q_f32(q3 + e);
    const float32x4_t qv4 = vld1q_f32(q4 + e);
    const float32x4_t qv5 = vld1q_f32(q5 + e);
    const float32x4_t qv6 = vld1q_f32(q6 + e);
    const float32x4_t qv7 = vld1q_f32(q7 + e);
    const float32x4_t qv8 = vld1q_f32(q8 + e);
    const float32x4_t qv9 = vld1q_f32(q9 + e);
    FUSED_CPP_QKT10_KSTEP(0);
    FUSED_CPP_QKT10_KSTEP(1);
    FUSED_CPP_QKT10_KSTEP(2);
    FUSED_CPP_QKT10_KSTEP(3);
  }
#undef FUSED_CPP_QKT10_KSTEP
#undef FUSED_CPP_QKT10_COL

  for (; e < E; ++e) {
    const float* kb = Kp + e * 8;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    const float f0 = q0[e], f1 = q1[e], f2 = q2[e], f3 = q3[e], f4 = q4[e];
    const float f5 = q5[e], f6 = q6[e], f7 = q7[e], f8 = q8[e], f9 = q9[e];
    a0c0 = vfmaq_n_f32(a0c0, b0, f0); a0c1 = vfmaq_n_f32(a0c1, b1, f0);
    a1c0 = vfmaq_n_f32(a1c0, b0, f1); a1c1 = vfmaq_n_f32(a1c1, b1, f1);
    a2c0 = vfmaq_n_f32(a2c0, b0, f2); a2c1 = vfmaq_n_f32(a2c1, b1, f2);
    a3c0 = vfmaq_n_f32(a3c0, b0, f3); a3c1 = vfmaq_n_f32(a3c1, b1, f3);
    a4c0 = vfmaq_n_f32(a4c0, b0, f4); a4c1 = vfmaq_n_f32(a4c1, b1, f4);
    a5c0 = vfmaq_n_f32(a5c0, b0, f5); a5c1 = vfmaq_n_f32(a5c1, b1, f5);
    a6c0 = vfmaq_n_f32(a6c0, b0, f6); a6c1 = vfmaq_n_f32(a6c1, b1, f6);
    a7c0 = vfmaq_n_f32(a7c0, b0, f7); a7c1 = vfmaq_n_f32(a7c1, b1, f7);
    a8c0 = vfmaq_n_f32(a8c0, b0, f8); a8c1 = vfmaq_n_f32(a8c1, b1, f8);
    a9c0 = vfmaq_n_f32(a9c0, b0, f9); a9c1 = vfmaq_n_f32(a9c1, b1, f9);
  }

  float32x4_t rows[10][2] = {
      {a0c0, a0c1}, {a1c0, a1c1}, {a2c0, a2c1}, {a3c0, a3c1}, {a4c0, a4c1},
      {a5c0, a5c1}, {a6c0, a6c1}, {a7c0, a7c1}, {a8c0, a8c1}, {a9c0, a9c1}};
  if (Sk == 8) {
    for (int r = 0; r < 10; ++r) {
      float* out = scores_buf + r * scores_row_stride;
      vst1q_f32(out + 0, rows[r][0]);
      vst1q_f32(out + 4, rows[r][1]);
    }
  } else {
    for (int r = 0; r < 10; ++r) {
      float tmp[8];
      vst1q_f32(tmp + 0, rows[r][0]);
      vst1q_f32(tmp + 4, rows[r][1]);
      float* out = scores_buf + r * scores_row_stride;
      for (int j = 0; j < Sk; ++j) out[j] = tmp[j];
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// Dispatch one tile: R rows (1..10) x Sk cols (1..8).
inline void qkt_10x8_tile(
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
    case 10: qkt_10x8_neon_h10(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 9: qkt_10x8_tile_neon<9>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 8: qkt_10x8_tile_neon<8>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 7: qkt_10x8_tile_neon<7>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 6: qkt_10x8_tile_neon<6>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 5: qkt_10x8_tile_neon<5>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 4: qkt_10x8_tile_neon<4>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 3: qkt_10x8_tile_neon<3>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 2: qkt_10x8_tile_neon<2>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    case 1: qkt_10x8_tile_neon<1>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, Sk); return;
    default: break;
  }
#endif
  qkt_tile_scalar(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, R, Sk);
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
// Row-max-fused QKT for a FULL 8-col S-block. Computes scores[R x 8] and folds
// the per-row running max into row_max[r] (kept in vector form; the caller
// reduces each to a scalar after the last S-block). Partial (Sk<8) blocks use
// the plain qkt_10x8_tile + a masked max in the orchestration.
template <int R>
inline void qkt_10x8_rowmax_tile_neon(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    float32x4_t row_max[10]) {
  static_assert(R >= 1 && R <= 10, "qkt_10x8_rowmax_tile_neon supports 1..10 rows");
  float32x4_t acc[R][2];
  for (int r = 0; r < R; ++r) {
    acc[r][0] = vdupq_n_f32(0.0f);
    acc[r][1] = vdupq_n_f32(0.0f);
  }
  const float* qr[R];
  for (int r = 0; r < R; ++r) {
    qr[r] = Q + r * q_row_stride;
  }

  int64_t e = 0;
#define FUSED_CPP_QKT10X8RM_KSTEP(KK)                                      \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 8;                                 \
    const float32x4_t b0 = vld1q_f32(kb + 0);                              \
    const float32x4_t b1 = vld1q_f32(kb + 4);                              \
    for (int r = 0; r < R; ++r) {                                          \
      acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, qv[r], (KK));             \
      acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, qv[r], (KK));             \
    }                                                                      \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    float32x4_t qv[R];
    for (int r = 0; r < R; ++r) {
      qv[r] = vld1q_f32(qr[r] + e);
    }
    FUSED_CPP_QKT10X8RM_KSTEP(0);
    FUSED_CPP_QKT10X8RM_KSTEP(1);
    FUSED_CPP_QKT10X8RM_KSTEP(2);
    FUSED_CPP_QKT10X8RM_KSTEP(3);
  }
#undef FUSED_CPP_QKT10X8RM_KSTEP

  for (; e < E; ++e) {
    const float* kb = Kp + e * 8;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    for (int r = 0; r < R; ++r) {
      const float qf = qr[r][e];
      acc[r][0] = vfmaq_n_f32(acc[r][0], b0, qf);
      acc[r][1] = vfmaq_n_f32(acc[r][1], b1, qf);
    }
  }

  for (int r = 0; r < R; ++r) {
    float* out = scores_buf + r * scores_row_stride;
    vst1q_f32(out + 0, acc[r][0]);
    vst1q_f32(out + 4, acc[r][1]);
    row_max[r] = vmaxq_f32(row_max[r], vmaxq_f32(acc[r][0], acc[r][1]));
  }
}

// Explicit 20-named-accumulator R=10 row-max-fused QKT (full 8-col block).
FUSED_CPP_PACK8_NOINLINE inline void qkt_10x8_rowmax_neon_h10(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    float32x4_t row_max[10]) {
  float32x4_t a0c0, a0c1, a1c0, a1c1, a2c0, a2c1, a3c0, a3c1, a4c0, a4c1,
      a5c0, a5c1, a6c0, a6c1, a7c0, a7c1, a8c0, a8c1, a9c0, a9c1;
  const float32x4_t z = vdupq_n_f32(0.0f);
  a0c0 = a0c1 = a1c0 = a1c1 = a2c0 = a2c1 = a3c0 = a3c1 = a4c0 = a4c1 =
      a5c0 = a5c1 = a6c0 = a6c1 = a7c0 = a7c1 = a8c0 = a8c1 = a9c0 = a9c1 = z;
  const float* q0 = Q; const float* q1 = Q + 1 * q_row_stride;
  const float* q2 = Q + 2 * q_row_stride; const float* q3 = Q + 3 * q_row_stride;
  const float* q4 = Q + 4 * q_row_stride; const float* q5 = Q + 5 * q_row_stride;
  const float* q6 = Q + 6 * q_row_stride; const float* q7 = Q + 7 * q_row_stride;
  const float* q8 = Q + 8 * q_row_stride; const float* q9 = Q + 9 * q_row_stride;

  int64_t e = 0;
  float32x4_t bA, bB;
#define FUSED_CPP_QKT10RM_COL(KK, COL, REG)                                \
  do {                                                                     \
    REG = vld1q_f32(kb + (COL) * 4);                                       \
    a0c##COL = vfmaq_laneq_f32(a0c##COL, REG, qv0, (KK));                  \
    a1c##COL = vfmaq_laneq_f32(a1c##COL, REG, qv1, (KK));                  \
    a2c##COL = vfmaq_laneq_f32(a2c##COL, REG, qv2, (KK));                  \
    a3c##COL = vfmaq_laneq_f32(a3c##COL, REG, qv3, (KK));                  \
    a4c##COL = vfmaq_laneq_f32(a4c##COL, REG, qv4, (KK));                  \
    a5c##COL = vfmaq_laneq_f32(a5c##COL, REG, qv5, (KK));                  \
    a6c##COL = vfmaq_laneq_f32(a6c##COL, REG, qv6, (KK));                  \
    a7c##COL = vfmaq_laneq_f32(a7c##COL, REG, qv7, (KK));                  \
    a8c##COL = vfmaq_laneq_f32(a8c##COL, REG, qv8, (KK));                  \
    a9c##COL = vfmaq_laneq_f32(a9c##COL, REG, qv9, (KK));                  \
  } while (0)
#define FUSED_CPP_QKT10RM_KSTEP(KK)                                        \
  do {                                                                     \
    const float* kb = Kp + (e + (KK)) * 8;                                 \
    FUSED_CPP_QKT10RM_COL(KK, 0, bA);                                      \
    FUSED_CPP_QKT10RM_COL(KK, 1, bB);                                      \
  } while (0)

  for (; e + 4 <= E; e += 4) {
    const float32x4_t qv0 = vld1q_f32(q0 + e);
    const float32x4_t qv1 = vld1q_f32(q1 + e);
    const float32x4_t qv2 = vld1q_f32(q2 + e);
    const float32x4_t qv3 = vld1q_f32(q3 + e);
    const float32x4_t qv4 = vld1q_f32(q4 + e);
    const float32x4_t qv5 = vld1q_f32(q5 + e);
    const float32x4_t qv6 = vld1q_f32(q6 + e);
    const float32x4_t qv7 = vld1q_f32(q7 + e);
    const float32x4_t qv8 = vld1q_f32(q8 + e);
    const float32x4_t qv9 = vld1q_f32(q9 + e);
    FUSED_CPP_QKT10RM_KSTEP(0);
    FUSED_CPP_QKT10RM_KSTEP(1);
    FUSED_CPP_QKT10RM_KSTEP(2);
    FUSED_CPP_QKT10RM_KSTEP(3);
  }
#undef FUSED_CPP_QKT10RM_KSTEP
#undef FUSED_CPP_QKT10RM_COL

  for (; e < E; ++e) {
    const float* kb = Kp + e * 8;
    const float32x4_t b0 = vld1q_f32(kb + 0);
    const float32x4_t b1 = vld1q_f32(kb + 4);
    const float f0 = q0[e], f1 = q1[e], f2 = q2[e], f3 = q3[e], f4 = q4[e];
    const float f5 = q5[e], f6 = q6[e], f7 = q7[e], f8 = q8[e], f9 = q9[e];
    a0c0 = vfmaq_n_f32(a0c0, b0, f0); a0c1 = vfmaq_n_f32(a0c1, b1, f0);
    a1c0 = vfmaq_n_f32(a1c0, b0, f1); a1c1 = vfmaq_n_f32(a1c1, b1, f1);
    a2c0 = vfmaq_n_f32(a2c0, b0, f2); a2c1 = vfmaq_n_f32(a2c1, b1, f2);
    a3c0 = vfmaq_n_f32(a3c0, b0, f3); a3c1 = vfmaq_n_f32(a3c1, b1, f3);
    a4c0 = vfmaq_n_f32(a4c0, b0, f4); a4c1 = vfmaq_n_f32(a4c1, b1, f4);
    a5c0 = vfmaq_n_f32(a5c0, b0, f5); a5c1 = vfmaq_n_f32(a5c1, b1, f5);
    a6c0 = vfmaq_n_f32(a6c0, b0, f6); a6c1 = vfmaq_n_f32(a6c1, b1, f6);
    a7c0 = vfmaq_n_f32(a7c0, b0, f7); a7c1 = vfmaq_n_f32(a7c1, b1, f7);
    a8c0 = vfmaq_n_f32(a8c0, b0, f8); a8c1 = vfmaq_n_f32(a8c1, b1, f8);
    a9c0 = vfmaq_n_f32(a9c0, b0, f9); a9c1 = vfmaq_n_f32(a9c1, b1, f9);
  }

  float32x4_t rows[10][2] = {
      {a0c0, a0c1}, {a1c0, a1c1}, {a2c0, a2c1}, {a3c0, a3c1}, {a4c0, a4c1},
      {a5c0, a5c1}, {a6c0, a6c1}, {a7c0, a7c1}, {a8c0, a8c1}, {a9c0, a9c1}};
  for (int r = 0; r < 10; ++r) {
    float* out = scores_buf + r * scores_row_stride;
    vst1q_f32(out + 0, rows[r][0]);
    vst1q_f32(out + 4, rows[r][1]);
    row_max[r] = vmaxq_f32(row_max[r], vmaxq_f32(rows[r][0], rows[r][1]));
  }
}

// Vector-interface row-max-fused dispatch: folds into row_max[10] (kept across
// S-blocks by the caller; reduced to scalar once after the last block).
inline void qkt_10x8_rowmax_vec(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int R,
    float32x4_t row_max[10]) {
  switch (R) {
    case 10: qkt_10x8_rowmax_neon_h10(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 9: qkt_10x8_rowmax_tile_neon<9>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 8: qkt_10x8_rowmax_tile_neon<8>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 7: qkt_10x8_rowmax_tile_neon<7>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 6: qkt_10x8_rowmax_tile_neon<6>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 5: qkt_10x8_rowmax_tile_neon<5>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 4: qkt_10x8_rowmax_tile_neon<4>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 3: qkt_10x8_rowmax_tile_neon<3>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 2: qkt_10x8_rowmax_tile_neon<2>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    case 1: qkt_10x8_rowmax_tile_neon<1>(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, row_max); return;
    default: return;
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// Dispatch a full-8 row-max-fused tile (R rows, Sk==8) with scalar interface.
inline void qkt_10x8_rowmax(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float* scores_buf,
    int64_t scores_row_stride,
    int R,
    float* row_max_scalar) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  float32x4_t rm[10];
  for (int r = 0; r < R; ++r) rm[r] = vdupq_n_f32(row_max_scalar[r]);
  qkt_10x8_rowmax_vec(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, R, rm);
  for (int r = 0; r < R; ++r) row_max_scalar[r] = vmaxvq_f32(rm[r]);
  return;
#else
  qkt_tile_scalar(Q, q_row_stride, Kp, E, scores_buf, scores_row_stride, R, 8);
  for (int r = 0; r < R; ++r) {
    float m = row_max_scalar[r];
    const float* out = scores_buf + r * scores_row_stride;
    for (int j = 0; j < 8; ++j) m = out[j] > m ? out[j] : m;
    row_max_scalar[r] = m;
  }
#endif
}

// ── PV 10x8 stream-B kernel ────────────────────────────────────────────────
//
// O[R x Ev_cur] += P[R x Sk] (P_row_stride) . Vp, where Vp is one Ev-block of
// packed V ([S][8], s-major). Reduction over the S-tile (k=0..Sk). P rows are
// held (lane-broadcast, unroll-4 over S); the 8 packed V cols are streamed.
// ACCUMULATES into O.

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
inline void pv_10x8_tile_neon(
    const float* P,
    int64_t P_row_stride,
    const float* Vp,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
    int64_t o_row_stride,
    int Ev_cur) {
  static_assert(R >= 1 && R <= 10, "pv_10x8_tile_neon supports 1..10 rows");
  float32x4_t acc[R][2];
  if (Ev_cur == 8) {
    for (int r = 0; r < R; ++r) {
      const float* o = O + r * o_row_stride;
      acc[r][0] = vld1q_f32(o + 0);
      acc[r][1] = vld1q_f32(o + 4);
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float tmp[8] = {0};
      const float* o = O + r * o_row_stride;
      for (int j = 0; j < Ev_cur; ++j) tmp[j] = o[j];
      acc[r][0] = vld1q_f32(tmp + 0);
      acc[r][1] = vld1q_f32(tmp + 4);
    }
  }

  const float* pr[R];
  for (int r = 0; r < R; ++r) {
    pr[r] = P + r * P_row_stride;
  }

  int64_t s = 0;
#define FUSED_CPP_PV10X8_VSTEP(KK)                                         \
  do {                                                                     \
    const float* vb = Vp + (s + (KK)) * v_row_stride;                      \
    const float32x4_t b0 = vld1q_f32(vb + 0);                              \
    const float32x4_t b1 = vld1q_f32(vb + 4);                              \
    for (int r = 0; r < R; ++r) {                                          \
      acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, pvv[r], (KK));            \
      acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, pvv[r], (KK));            \
    }                                                                      \
  } while (0)

  for (; s + 4 <= Sk; s += 4) {
    float32x4_t pvv[R];
    for (int r = 0; r < R; ++r) {
      pvv[r] = vld1q_f32(pr[r] + s);
    }
    FUSED_CPP_PV10X8_VSTEP(0);
    FUSED_CPP_PV10X8_VSTEP(1);
    FUSED_CPP_PV10X8_VSTEP(2);
    FUSED_CPP_PV10X8_VSTEP(3);
  }
#undef FUSED_CPP_PV10X8_VSTEP

  for (; s < Sk; ++s) {
    const float* vb = Vp + s * v_row_stride;
    const float32x4_t b0 = vld1q_f32(vb + 0);
    const float32x4_t b1 = vld1q_f32(vb + 4);
    for (int r = 0; r < R; ++r) {
      const float pf = pr[r][s];
      acc[r][0] = vfmaq_n_f32(acc[r][0], b0, pf);
      acc[r][1] = vfmaq_n_f32(acc[r][1], b1, pf);
    }
  }

  if (Ev_cur == 8) {
    for (int r = 0; r < R; ++r) {
      float* o = O + r * o_row_stride;
      vst1q_f32(o + 0, acc[r][0]);
      vst1q_f32(o + 4, acc[r][1]);
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float tmp[8];
      vst1q_f32(tmp + 0, acc[r][0]);
      vst1q_f32(tmp + 4, acc[r][1]);
      float* o = O + r * o_row_stride;
      for (int j = 0; j < Ev_cur; ++j) o[j] = tmp[j];
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
// Explicit 20-named-accumulator R=10 PV fast path (accumulate into O). 10 held P
// rows (pv0..pv9, 4 S-values each), V streamed via 2 regs (bA/bB).
FUSED_CPP_PACK8_NOINLINE inline void pv_10x8_neon_h10(
    const float* P,
    int64_t P_row_stride,
    const float* Vp,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
    int64_t o_row_stride,
    int Ev_cur) {
  float32x4_t a0c0, a0c1, a1c0, a1c1, a2c0, a2c1, a3c0, a3c1, a4c0, a4c1,
      a5c0, a5c1, a6c0, a6c1, a7c0, a7c1, a8c0, a8c1, a9c0, a9c1;
  if (Ev_cur == 8) {
    const float* o0 = O; const float* o1 = O + o_row_stride;
    const float* o2 = O + 2 * o_row_stride; const float* o3 = O + 3 * o_row_stride;
    const float* o4 = O + 4 * o_row_stride; const float* o5 = O + 5 * o_row_stride;
    const float* o6 = O + 6 * o_row_stride; const float* o7 = O + 7 * o_row_stride;
    const float* o8 = O + 8 * o_row_stride; const float* o9 = O + 9 * o_row_stride;
    a0c0 = vld1q_f32(o0 + 0); a0c1 = vld1q_f32(o0 + 4);
    a1c0 = vld1q_f32(o1 + 0); a1c1 = vld1q_f32(o1 + 4);
    a2c0 = vld1q_f32(o2 + 0); a2c1 = vld1q_f32(o2 + 4);
    a3c0 = vld1q_f32(o3 + 0); a3c1 = vld1q_f32(o3 + 4);
    a4c0 = vld1q_f32(o4 + 0); a4c1 = vld1q_f32(o4 + 4);
    a5c0 = vld1q_f32(o5 + 0); a5c1 = vld1q_f32(o5 + 4);
    a6c0 = vld1q_f32(o6 + 0); a6c1 = vld1q_f32(o6 + 4);
    a7c0 = vld1q_f32(o7 + 0); a7c1 = vld1q_f32(o7 + 4);
    a8c0 = vld1q_f32(o8 + 0); a8c1 = vld1q_f32(o8 + 4);
    a9c0 = vld1q_f32(o9 + 0); a9c1 = vld1q_f32(o9 + 4);
  } else {
    float tmp[10][8] = {{0}};
    for (int r = 0; r < 10; ++r)
      for (int j = 0; j < Ev_cur; ++j) tmp[r][j] = O[r * o_row_stride + j];
    a0c0 = vld1q_f32(tmp[0] + 0); a0c1 = vld1q_f32(tmp[0] + 4);
    a1c0 = vld1q_f32(tmp[1] + 0); a1c1 = vld1q_f32(tmp[1] + 4);
    a2c0 = vld1q_f32(tmp[2] + 0); a2c1 = vld1q_f32(tmp[2] + 4);
    a3c0 = vld1q_f32(tmp[3] + 0); a3c1 = vld1q_f32(tmp[3] + 4);
    a4c0 = vld1q_f32(tmp[4] + 0); a4c1 = vld1q_f32(tmp[4] + 4);
    a5c0 = vld1q_f32(tmp[5] + 0); a5c1 = vld1q_f32(tmp[5] + 4);
    a6c0 = vld1q_f32(tmp[6] + 0); a6c1 = vld1q_f32(tmp[6] + 4);
    a7c0 = vld1q_f32(tmp[7] + 0); a7c1 = vld1q_f32(tmp[7] + 4);
    a8c0 = vld1q_f32(tmp[8] + 0); a8c1 = vld1q_f32(tmp[8] + 4);
    a9c0 = vld1q_f32(tmp[9] + 0); a9c1 = vld1q_f32(tmp[9] + 4);
  }
  const float* p0 = P; const float* p1 = P + 1 * P_row_stride;
  const float* p2 = P + 2 * P_row_stride; const float* p3 = P + 3 * P_row_stride;
  const float* p4 = P + 4 * P_row_stride; const float* p5 = P + 5 * P_row_stride;
  const float* p6 = P + 6 * P_row_stride; const float* p7 = P + 7 * P_row_stride;
  const float* p8 = P + 8 * P_row_stride; const float* p9 = P + 9 * P_row_stride;

  int64_t s = 0;
  float32x4_t bA, bB;
#define FUSED_CPP_PV10_COL(KK, COL, REG)                                   \
  do {                                                                     \
    REG = vld1q_f32(vb + (COL) * 4);                                       \
    a0c##COL = vfmaq_laneq_f32(a0c##COL, REG, pv0, (KK));                  \
    a1c##COL = vfmaq_laneq_f32(a1c##COL, REG, pv1, (KK));                  \
    a2c##COL = vfmaq_laneq_f32(a2c##COL, REG, pv2, (KK));                  \
    a3c##COL = vfmaq_laneq_f32(a3c##COL, REG, pv3, (KK));                  \
    a4c##COL = vfmaq_laneq_f32(a4c##COL, REG, pv4, (KK));                  \
    a5c##COL = vfmaq_laneq_f32(a5c##COL, REG, pv5, (KK));                  \
    a6c##COL = vfmaq_laneq_f32(a6c##COL, REG, pv6, (KK));                  \
    a7c##COL = vfmaq_laneq_f32(a7c##COL, REG, pv7, (KK));                  \
    a8c##COL = vfmaq_laneq_f32(a8c##COL, REG, pv8, (KK));                  \
    a9c##COL = vfmaq_laneq_f32(a9c##COL, REG, pv9, (KK));                  \
  } while (0)
#define FUSED_CPP_PV10_SSTEP(KK)                                           \
  do {                                                                     \
    const float* vb = Vp + (s + (KK)) * v_row_stride;                      \
    FUSED_CPP_PV10_COL(KK, 0, bA);                                         \
    FUSED_CPP_PV10_COL(KK, 1, bB);                                         \
  } while (0)

  for (; s + 4 <= Sk; s += 4) {
    const float32x4_t pv0 = vld1q_f32(p0 + s);
    const float32x4_t pv1 = vld1q_f32(p1 + s);
    const float32x4_t pv2 = vld1q_f32(p2 + s);
    const float32x4_t pv3 = vld1q_f32(p3 + s);
    const float32x4_t pv4 = vld1q_f32(p4 + s);
    const float32x4_t pv5 = vld1q_f32(p5 + s);
    const float32x4_t pv6 = vld1q_f32(p6 + s);
    const float32x4_t pv7 = vld1q_f32(p7 + s);
    const float32x4_t pv8 = vld1q_f32(p8 + s);
    const float32x4_t pv9 = vld1q_f32(p9 + s);
    FUSED_CPP_PV10_SSTEP(0);
    FUSED_CPP_PV10_SSTEP(1);
    FUSED_CPP_PV10_SSTEP(2);
    FUSED_CPP_PV10_SSTEP(3);
  }
#undef FUSED_CPP_PV10_SSTEP
#undef FUSED_CPP_PV10_COL

  for (; s < Sk; ++s) {
    const float* vb = Vp + s * v_row_stride;
    const float32x4_t b0 = vld1q_f32(vb + 0);
    const float32x4_t b1 = vld1q_f32(vb + 4);
    const float f0 = p0[s], f1 = p1[s], f2 = p2[s], f3 = p3[s], f4 = p4[s];
    const float f5 = p5[s], f6 = p6[s], f7 = p7[s], f8 = p8[s], f9 = p9[s];
    a0c0 = vfmaq_n_f32(a0c0, b0, f0); a0c1 = vfmaq_n_f32(a0c1, b1, f0);
    a1c0 = vfmaq_n_f32(a1c0, b0, f1); a1c1 = vfmaq_n_f32(a1c1, b1, f1);
    a2c0 = vfmaq_n_f32(a2c0, b0, f2); a2c1 = vfmaq_n_f32(a2c1, b1, f2);
    a3c0 = vfmaq_n_f32(a3c0, b0, f3); a3c1 = vfmaq_n_f32(a3c1, b1, f3);
    a4c0 = vfmaq_n_f32(a4c0, b0, f4); a4c1 = vfmaq_n_f32(a4c1, b1, f4);
    a5c0 = vfmaq_n_f32(a5c0, b0, f5); a5c1 = vfmaq_n_f32(a5c1, b1, f5);
    a6c0 = vfmaq_n_f32(a6c0, b0, f6); a6c1 = vfmaq_n_f32(a6c1, b1, f6);
    a7c0 = vfmaq_n_f32(a7c0, b0, f7); a7c1 = vfmaq_n_f32(a7c1, b1, f7);
    a8c0 = vfmaq_n_f32(a8c0, b0, f8); a8c1 = vfmaq_n_f32(a8c1, b1, f8);
    a9c0 = vfmaq_n_f32(a9c0, b0, f9); a9c1 = vfmaq_n_f32(a9c1, b1, f9);
  }

  float32x4_t rows[10][2] = {
      {a0c0, a0c1}, {a1c0, a1c1}, {a2c0, a2c1}, {a3c0, a3c1}, {a4c0, a4c1},
      {a5c0, a5c1}, {a6c0, a6c1}, {a7c0, a7c1}, {a8c0, a8c1}, {a9c0, a9c1}};
  if (Ev_cur == 8) {
    for (int r = 0; r < 10; ++r) {
      float* o = O + r * o_row_stride;
      vst1q_f32(o + 0, rows[r][0]);
      vst1q_f32(o + 4, rows[r][1]);
    }
  } else {
    for (int r = 0; r < 10; ++r) {
      float tmp[8];
      vst1q_f32(tmp + 0, rows[r][0]);
      vst1q_f32(tmp + 4, rows[r][1]);
      float* o = O + r * o_row_stride;
      for (int j = 0; j < Ev_cur; ++j) o[j] = tmp[j];
    }
  }
}
#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// Dispatch one PV tile: R rows (1..10) x Ev_cur cols (1..8), reduction Sk.
inline void pv_10x8_tile(
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
    case 10: pv_10x8_neon_h10(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 9: pv_10x8_tile_neon<9>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 8: pv_10x8_tile_neon<8>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 7: pv_10x8_tile_neon<7>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 6: pv_10x8_tile_neon<6>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 5: pv_10x8_tile_neon<5>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 4: pv_10x8_tile_neon<4>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 3: pv_10x8_tile_neon<3>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 2: pv_10x8_tile_neon<2>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    case 1: pv_10x8_tile_neon<1>(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, Ev_cur); return;
    default: break;
  }
#endif
  pv_tile_scalar_acc(P, P_row_stride, Vp, v_row_stride, Sk, O, o_row_stride, R, Ev_cur);
}

}  // namespace sdpa_10x8
}  // namespace fused_cpp
