#pragma once
// Minimal SDPA parameter definitions for the extracted fp32 packqkv kernel.
// This standalone copy intentionally omits the multi-version registry used by
// the repository-level framework.

#include <cstddef>
#include <cstdint>

// ── SDPA dtype 标签 ──────────────────────────────────────────────────────
//
// 直接以整型 enum 表示，避免在头文件中引入外部框架依赖。
enum class SdpaDtype : int32_t {
    kFloat32  = 0,
    kBFloat16 = 1,
};

// ── SDPA 参数结构体 ──────────────────────────────────────────────────────
struct SdpaParams {
    // ── 形状 ──
    int64_t B   = 0;   ///< batch_size
    int64_t N   = 0;   ///< num_heads
    int64_t L   = 0;   ///< query seq_len
    int64_t S   = 0;   ///< key/value seq_len
    int64_t E   = 0;   ///< qk_head_dim
    int64_t Ev  = 0;   ///< v_head_dim

    // ── 标量/掩码控制 ──
    float scale_f       = 0.0f;             ///< 缩放因子
    float neg_inf       = 0.0f;             ///< -infinity 占位
    int64_t causal_offset = 0;              ///< S - L
    bool is_causal      = false;
    SdpaDtype dtype     = SdpaDtype::kFloat32;

    // ── 数据指针（dtype-erased）──
    // q_ptr / k_ptr / v_ptr 的实际类型由 dtype 决定：
    //   dtype == kFloat32  -> const float*
    //   dtype == kBFloat16 -> const at::BFloat16* from the local shim
    //                         (binary-equivalent to uint16_t)
    const void* q_ptr   = nullptr;
    const void* k_ptr   = nullptr;
    const void* v_ptr   = nullptr;

    // mask 指针始终是 fp32 contiguous [B,N,L,S]（或 nullptr）
    const float* mask_ptr = nullptr;

    // Optional ggml additive masks [S,L,H_mask,B_mask]. When either is set,
    // kernels read it directly with byte strides and broadcast H/B by modulo.
    const uint16_t* mask_f16_ptr = nullptr;
    int64_t mask_f16_ne0 = 0;
    int64_t mask_f16_ne1 = 0;
    int64_t mask_f16_ne2 = 0;
    int64_t mask_f16_ne3 = 0;
    int64_t mask_f16_nb0 = 0;
    int64_t mask_f16_nb1 = 0;
    int64_t mask_f16_nb2 = 0;
    int64_t mask_f16_nb3 = 0;
    const float* mask_f32_ptr = nullptr;
    int64_t mask_f32_ne0 = 0;
    int64_t mask_f32_ne1 = 0;
    int64_t mask_f32_ne2 = 0;
    int64_t mask_f32_ne3 = 0;
    int64_t mask_f32_nb0 = 0;
    int64_t mask_f32_nb1 = 0;
    int64_t mask_f32_nb2 = 0;
    int64_t mask_f32_nb3 = 0;

    // 输出指针始终为 fp32 累加缓冲
    float* out_ptr      = nullptr;

    // Optional fp32 statistics output in contiguous [B, N, L] layout.
    // Kernels that do not produce these values may leave them untouched.
    float* max_logits_ptr = nullptr;
    float* lse_ptr        = nullptr;
};
