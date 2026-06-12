#pragma once
// ── Shared feature configuration for flash2_neon_cache micro-kernels ────────

#if defined(__aarch64__)
#include <arm_neon.h>
#ifndef FUSED_CPP_SDPA_CACHE_HAS_NEON
#define FUSED_CPP_SDPA_CACHE_HAS_NEON 1
#endif
#else
#ifndef FUSED_CPP_SDPA_CACHE_HAS_NEON
#define FUSED_CPP_SDPA_CACHE_HAS_NEON 0
#endif
#endif

#ifndef FUSED_CPP_SDPA_CACHE_HAS_SVE
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#define FUSED_CPP_SDPA_CACHE_HAS_SVE 1
#else
#define FUSED_CPP_SDPA_CACHE_HAS_SVE 0
#endif
#endif

#ifndef FUSED_CPP_SDPA_CACHE_HAS_BF16
// 不同 GCC 版本的 ACLE 实现有差异：
//   * GCC 13+ / Clang：定义聚合宏 ``__ARM_FEATURE_BF16``。
//   * GCC 10–12（含鲲鹏 RH/CentOS 8 上常见的 gcc-toolset-11/12）：
//     **不**定义聚合宏，仅定义细分宏
//     ``__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`` /
//     ``__ARM_FEATURE_BF16_SCALAR_ARITHMETIC``。
// 本仓库实际依赖的是 NEON BF16 指令（vbfdotq / vbfmmlaq / vbfmlalbq /
// vbfmlaltq），它们由 vector arithmetic 子特性提供，因此任一定义都意味着
// 平台支持本文件覆盖的 bf16 路径，必须 OR 起来。
#if FUSED_CPP_SDPA_CACHE_HAS_NEON && \
    (defined(__ARM_FEATURE_BF16) || \
     defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
#define FUSED_CPP_SDPA_CACHE_HAS_BF16 1
#else
#define FUSED_CPP_SDPA_CACHE_HAS_BF16 0
#endif
#endif

#ifndef FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
// BFMMLA 的 NEON 形式（vbfmmlaq_f32）属于 FEAT_BF16，与 BFDOT / BFMLAL
// 同级；它**不**依赖 SVE 的 ``__ARM_FEATURE_MATMUL_FP`` —— 后者只在 SVE
// F32MM/F64MM 平台（Neoverse V1/V2 等）定义。鲲鹏 9000 等纯 NEON+BF16
// 平台也提供 BFMMLA，必须用更宽松的判定。
// Apple Silicon 仍然走旧分支：Apple Clang 不定义 ``__ARM_FEATURE_MATMUL_FP``
// 也不定义 ``__ARM_FEATURE_BF16``，但芯片有 BFMMLA。
#if FUSED_CPP_SDPA_CACHE_HAS_BF16 || defined(__APPLE__)
#define FUSED_CPP_SDPA_CACHE_HAS_BFMMLA 1
#else
#define FUSED_CPP_SDPA_CACHE_HAS_BFMMLA 0
#endif
#endif

#ifndef FUSED_CPP_SDPA_BF16_USE_BFMMLA
#define FUSED_CPP_SDPA_BF16_USE_BFMMLA 1
#endif
#ifndef FUSED_CPP_SDPA_BF16_USE_BFMLAL
#define FUSED_CPP_SDPA_BF16_USE_BFMLAL 1
#endif
#ifndef FUSED_CPP_SDPA_BF16_USE_DOT
#define FUSED_CPP_SDPA_BF16_USE_DOT 0
#endif

#if FUSED_CPP_SDPA_BF16_USE_DOT
static_assert(FUSED_CPP_SDPA_BF16_USE_BFMMLA == 0,
              "FUSED_CPP_SDPA_BF16_USE_DOT and FUSED_CPP_SDPA_BF16_USE_BFMMLA "
              "are mutually exclusive");
static_assert(FUSED_CPP_SDPA_BF16_USE_BFMLAL == 0,
              "FUSED_CPP_SDPA_BF16_USE_DOT and FUSED_CPP_SDPA_BF16_USE_BFMLAL "
              "are mutually exclusive");
#endif

#ifndef FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA
#if FUSED_CPP_SDPA_BF16_USE_BFMMLA && FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA 1
#else
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA 0
#endif
#endif

#ifndef FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMLAL
#if !FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA && \
    FUSED_CPP_SDPA_BF16_USE_BFMLAL && FUSED_CPP_SDPA_CACHE_HAS_BF16
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMLAL 1
#else
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMLAL 0
#endif
#endif

#ifndef FUSED_CPP_SDPA_CACHE_BF16_PATH_BFDOT
#if FUSED_CPP_SDPA_BF16_USE_DOT && FUSED_CPP_SDPA_CACHE_HAS_BF16
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_BFDOT 1
#else
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_BFDOT 0
#endif
#endif

#ifndef FUSED_CPP_SDPA_CACHE_BF16_PATH_WIDEN_FMLA
#if !FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA && \
    !FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMLAL && \
    !FUSED_CPP_SDPA_CACHE_BF16_PATH_BFDOT
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_WIDEN_FMLA 1
#else
#define FUSED_CPP_SDPA_CACHE_BF16_PATH_WIDEN_FMLA 0
#endif
#endif
