#pragma once
// ── 微内核 impl：MK_QkPackqkSeq4BmajorPvPquad ───────────────────────────────
//
// 组合 trait：QKᵀ 走 MK_QkPackqkSeq4Bmajor（Q+K 双 pack + B-major BFMMLA
// 调度 + 4 条独立 vld1q_u16），PV 走 MK_PQuad（fp32 P-quad；bf16
// P->bf16 + BFMLAL）。其余 op 与 baseline 一致。
//
// 设计动机：
//   * 在 bf16 SDPA 主路径，目前两种最优实现互斥——选 packqk_seq4_bmajor
//     拿到 bf16 QKᵀ 1.07× 加速，但 fp32 PV 退回 baseline；选 pquad 拿
//     到 fp32 PV 1.10× 加速，但 bf16 QKᵀ 退回 baseline。组合 trait 把
//     两个独立维度的优势合并，QKᵀ/PV 各取其最优实现。
//
// 与 MK_QkPackqkSeq4Bmajor 的唯一差别：
//   * fp32 版本的 pv_8x8 改派到 gemm_pv_microkernel_8x8_fp32_pquad
//     （而不是 baseline 的 gemm_pv_8x8 fp32 路径）。
//
// 与 MK_PQuad 的差别：
//   * bf16 qkt_8x8 改派到 packqk_seq4_bmajor inner（带 Q/K 双 pack 的
//     thread_local cache）。
//
// 数值上：
//   * bf16 QKᵀ：BFMMLA 累加顺序与 baseline 略有差异（B-major 调度），
//     按位不等价，与 `packqk_seq4_bmajor` 一致（已在 microkernel 单测
//     中按 atol/rtol 验证）。
//   * fp32 PV：与 baseline fp32 PV 按位等价（与 `pquad` 一致）。
//   * bf16 PV：P 临时 fp32→bf16 round 后用 BFMLALB/T；非按位等价，
//     但在 bf16 SDPA / microkernel 容差内。
//
// thread_local cache 复用 packqk_seq4_bmajor 的策略（Q / K 独立 buffer）。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD，默认 1。

#include "sdpa_standalone_shim.h"
#include <cstdint>
#include <vector>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD
#define FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD
struct MK_QkPackqkSeq4BmajorPvPquad {
  static constexpr const char* kName = "qk_packqk_seq4_bmajor_pv_pquad";
  static constexpr bool kEnabled = true;
  static constexpr bool kHasPvPbf16 = true;

  // —— QKᵀ 主体 8×8 bf16：走 packqk_seq4_bmajor（Q+K 双 pack + B-major）——
  static inline void qkt_8x8(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
    // ── Q 侧 thread_local cache（与 MK_QkPackqkSeq4Bmajor 独立一份）──
    // 不与其它 trait 共用 buffer：两个 trait 的 cache 失效策略相同，
    // 但 buffer 共用会让独立判断失效。
    static thread_local std::vector<uint16_t> q_packed_buf;
    static thread_local const at::BFloat16* last_Q = nullptr;
    static thread_local int64_t last_q_row_stride = 0;
    static thread_local int64_t last_q_E = 0;

    if (Q != last_Q || q_row_stride != last_q_row_stride || E != last_q_E) {
      q_packed_buf.assign(static_cast<size_t>(8 * E), 0);
      pack_q_8rows_to_seq_bf16(Q, q_row_stride, E, q_packed_buf.data());
      last_Q = Q;
      last_q_row_stride = q_row_stride;
      last_q_E = E;
    }

    // ── K 侧 thread_local cache ──
    static thread_local std::vector<uint16_t> k_packed_buf;
    static thread_local const at::BFloat16* last_K = nullptr;
    static thread_local int64_t last_k_row_stride = 0;
    static thread_local int64_t last_k_E = 0;

    if (K != last_K || k_row_stride != last_k_row_stride || E != last_k_E) {
      k_packed_buf.assign(static_cast<size_t>(8 * E), 0);
      pack_k_8rows_to_seq_bf16(K, k_row_stride, E, k_packed_buf.data());
      last_K = K;
      last_k_row_stride = k_row_stride;
      last_k_E = E;
    }

    gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner(
        q_packed_buf.data(), Q, q_row_stride,
        k_packed_buf.data(), K, k_row_stride,
        E, scale, scores_buf);
#else
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
  }
  // fp32 QKᵀ：与 baseline 同。
  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }

  // —— QKᵀ 退化 8×4（与 baseline 同） ——
  static inline void qkt_8x4(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale,
                 scores_buf, scores_row_stride);
  }
  static inline void qkt_8x4(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale,
                 scores_buf, scores_row_stride);
  }

  // —— QKᵀ 任意尾部（与 baseline 同） ——
  static inline void qkt_tail(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale,
                  scores_buf, scores_row_stride, Lq, Sk);
  }
  static inline void qkt_tail(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale,
                  scores_buf, scores_row_stride, Lq, Sk);
  }

  // —— P̂·V 主体 8×8 ——
  // bf16：派到 _bf16_pquad（与 MK_PQuad 一致；BF16 目标走 P->bf16 BFMLAL）。
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    gemm_pv_microkernel_8x8_bf16_pquad(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }
  // fp32：派到 _pquad 微内核（与 MK_PQuad 一致）。
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    gemm_pv_microkernel_8x8_fp32_pquad(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }

  // —— P̂ 已经是 bf16 的 PV 主体 8×8（microkernel 上限评估） ——
  static inline void pv_8x8_pbf16(
      const at::BFloat16* P_bf16, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    gemm_pv_microkernel_8x8_bf16_pbf16_prepacked(
        P_bf16, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }

  // —— P̂·V 任意尾部（与 baseline 同） ——
  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk,
                 O, o_row_stride, Lq, Ev);
  }
  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk,
                 O, o_row_stride, Lq, Ev);
  }
};
#endif  // FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD

}  // namespace fused_cpp::sdpa_microkernels
