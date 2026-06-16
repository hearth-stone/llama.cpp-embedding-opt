#pragma once
// ── L3-Resident FlashAttention-2 SDPA 内层模板（impl header）─────────────
//
// 这个文件持有 `process_q_tile_lc` / `run_path_collapse3` /
// `run_path_taskloop` 三个模板，原本住在 sdpa_flash2_neon_l3kv.cpp 里。
// 抽到 header 是为了让两个 SDPA 变体共用同一套模板：
//
//   * `flash2_neon_l3kv`        —— 直接用原始 V tensor（kPackedV=false）；
//   * `flash2_neon_l3kv_packv`  —— 入口处先把 V 重排成 [B, N, Ev/8, S, 8]
//                                  连续 layout，内层用 `kPackedV=true`
//                                  让微内核以 v_row_stride=8 访问 V。
//
// `bool kPackedV` 模板参数通过 `if constexpr` 在 V 指针计算处分两条路径，
// 默认值 false 保持原 l3kv.cpp 的行为。`v_evblock_stride` 是新增参数，
// 仅在 packed 路径下有意义（值 = S * 8）；非 packed 路径下传 0、不读。
//
// 头文件 inline 模板、不暴露符号；两个 .cpp 各自实例化 fully-specialized
// SDPA，热路径无运行期间接跳转。

#include "sdpa_standalone_shim.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "sdpa_common.h"
#include "sdpa_profile.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sdpa_microkernels/neon_cache_config.h"
#include "sdpa_tile_sizes.h"
#include "sdpa_microkernels/mk_traits.h"
#include "sdpa_pack_utils.h"

namespace fused_cpp::sdpa_flash2_neon_l3kv_impl {

using ::fused_cpp::sdpa_tile_sizes::TileSizes;
using ::fused_cpp::sdpa_tile_sizes::ceil_div_pos;

template <class MK, typename scalar_t, class = void>
struct has_k_sblock4_layout : std::false_type {};

template <class MK, typename scalar_t>
struct has_k_sblock4_layout<
    MK,
    scalar_t,
    std::void_t<decltype(MK::kHasKSBlock4Layout)>>
    : std::bool_constant<MK::kHasKSBlock4Layout> {};

template <class MK, typename scalar_t>
inline constexpr bool has_k_sblock4_layout_v =
    has_k_sblock4_layout<MK, scalar_t>::value;

template <class MK, typename scalar_t>
inline const scalar_t* k_tile_ptr_impl(
    const scalar_t* k_base,
    int64_t s_global,
    int64_t k_stride_s,
    int64_t E) {
  if constexpr (has_k_sblock4_layout_v<MK, scalar_t>) {
    return k_base + (s_global / 4) * E * 4 + (s_global & 3);
  } else {
    return k_base + s_global * k_stride_s;
  }
}

// ──────────────────────────────────────────────────────────────────────
// 软件预取与向量化 helper（与原 sdpa_flash2_neon_l3kv.cpp 中 helper 等价；
// 这里直接复用 namespace 内的实现，外层 .cpp 只 include 本 header）。
// ──────────────────────────────────────────────────────────────────────

#ifndef FUSED_CPP_SDPA_DISABLE_PREFETCH
#define FUSED_CPP_SDPA_DISABLE_PREFETCH 0
#endif

#ifndef FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE
#define FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE 4
#endif

static_assert(
    FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE == 4 ||
    FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE == 5 ||
    FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE == 6,
    "FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE must be 4, 5, or 6");

inline void prefetch_l1_keep_impl(const void* addr) {
#if !FUSED_CPP_SDPA_DISABLE_PREFETCH
  __builtin_prefetch(addr, 0 /*read*/, 3 /*L1 keep*/);
#else
  (void)addr;
#endif
}

inline void prefetch_l2_keep_impl(const void* addr) {
#if !FUSED_CPP_SDPA_DISABLE_PREFETCH
  __builtin_prefetch(addr, 0 /*read*/, 2 /*L2 keep*/);
#else
  (void)addr;
#endif
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON

// vexpq_f32：基于 range reduction + 多项式估值的 NEON 向量化 expf 近似。
template <int kDegree = FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE>
static inline float32x4_t vexpq_f32_poly_impl(float32x4_t x) {
  static_assert(kDegree == 4 || kDegree == 5 || kDegree == 6,
                "vexpq_f32_poly_impl supports degree 4, 5, or 6");
  const float32x4_t kLn2  = vdupq_n_f32(0.6931471805599453f);
  const float32x4_t kInvLn2 = vdupq_n_f32(1.4426950408889634f);
  const float32x4_t c0 = vdupq_n_f32(1.0f);
  const float32x4_t c1 = vdupq_n_f32(1.0f);
  const float32x4_t c2 = vdupq_n_f32(0.5f);
  const float32x4_t c3 = vdupq_n_f32(0.16666666f);
  const float32x4_t c4 = vdupq_n_f32(0.04166666f);
  const float32x4_t c5 = vdupq_n_f32(0.00833333f);
  const float32x4_t c6 = vdupq_n_f32(0.0013888889f);

  const float32x4_t kLo = vdupq_n_f32(-87.0f);
  // softmax inputs are x = score - row_max <= 0, so the upper clamp at +87 is
  // never reached; keep only the lower clamp (maps masked -inf lanes to ~0).
  x = vmaxq_f32(x, kLo);

  float32x4_t fn = vrndnq_f32(vmulq_f32(x, kInvLn2));
  int32x4_t n = vcvtq_s32_f32(fn);

  float32x4_t r = vfmsq_f32(x, fn, kLn2);

  float32x4_t poly;
  if constexpr (kDegree == 4) {
    poly = c4;
  } else if constexpr (kDegree == 5) {
    poly = c5;
    poly = vfmaq_f32(c4, poly, r);
  } else {
    poly = c6;
    poly = vfmaq_f32(c5, poly, r);
    poly = vfmaq_f32(c4, poly, r);
  }
  poly = vfmaq_f32(c3, poly, r);
  poly = vfmaq_f32(c2, poly, r);
  poly = vfmaq_f32(c1, poly, r);
  poly = vfmaq_f32(c0, poly, r);

  int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);

  return vmulq_f32(poly, pow2n);
}

static inline float32x4_t vexpq_f32_impl(float32x4_t x) {
  return vexpq_f32_poly_impl<FUSED_CPP_SDPA_SOFTMAX_EXP_POLY_DEGREE>(x);
}

#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

inline float vectorized_exp_minus_impl(
    float* dst, const float* src, float new_max, int64_t len) {
  float block_sum = 0.0f;
  int64_t j = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  const float32x4_t vmax = vdupq_n_f32(new_max);
  float32x4_t vsum = vdupq_n_f32(0.0f);
  float32x4_t vsum1 = vdupq_n_f32(0.0f);
  for (; j + 8 <= len; j += 8) {
    float32x4_t s0 = vld1q_f32(src + j);
    float32x4_t s1 = vld1q_f32(src + j + 4);
    float32x4_t e0 = vexpq_f32_impl(vsubq_f32(s0, vmax));
    float32x4_t e1 = vexpq_f32_impl(vsubq_f32(s1, vmax));
    vst1q_f32(dst + j, e0);
    vst1q_f32(dst + j + 4, e1);
    vsum = vaddq_f32(vsum, e0);
    vsum1 = vaddq_f32(vsum1, e1);
  }
  for (; j + 4 <= len; j += 4) {
    float32x4_t s0 = vld1q_f32(src + j);
    float32x4_t e0 = vexpq_f32_impl(vsubq_f32(s0, vmax));
    vst1q_f32(dst + j, e0);
    vsum = vaddq_f32(vsum, e0);
  }
  block_sum = vaddvq_f32(vaddq_f32(vsum, vsum1));
#endif
  for (; j < len; ++j) {
    float e = std::exp(src[j] - new_max);
    dst[j] = e;
    block_sum += e;
  }
  return block_sum;
}

inline float max_update_impl(float current, const float* src, int64_t len) {
  int64_t j = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  float32x4_t vm = vdupq_n_f32(current);
  float32x4_t vm1 = vdupq_n_f32(current);
  for (; j + 8 <= len; j += 8) {
    vm = vmaxq_f32(vm, vld1q_f32(src + j));
    vm1 = vmaxq_f32(vm1, vld1q_f32(src + j + 4));
  }
  for (; j + 4 <= len; j += 4) {
    vm = vmaxq_f32(vm, vld1q_f32(src + j));
  }
  current = vmaxvq_f32(vmaxq_f32(vm, vm1));
#endif
  for (; j < len; ++j) {
    if (src[j] > current) current = src[j];
  }
  return current;
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON
inline void update_row_max_block_neon_impl(
    float32x4_t row_max_lo[8],
    float32x4_t row_max_hi[8],
    float row_max_tail[8],
    const float* scores,
    int64_t scores_row_stride,
    int l_count,
    int s_count) {
  for (int i = 0; i < l_count; ++i) {
    const float* row = scores + i * scores_row_stride;
    int j = 0;
    for (; j + 8 <= s_count; j += 8) {
      row_max_lo[i] = vmaxq_f32(row_max_lo[i], vld1q_f32(row + j));
      row_max_hi[i] = vmaxq_f32(row_max_hi[i], vld1q_f32(row + j + 4));
    }
    for (; j + 4 <= s_count; j += 4) {
      row_max_lo[i] = vmaxq_f32(row_max_lo[i], vld1q_f32(row + j));
    }
    if (j < s_count) {
      row_max_tail[i] =
          max_update_impl(row_max_tail[i], row + j, s_count - j);
    }
  }
}
#endif

inline void update_row_max_block_impl(
    float row_max[8],
    const float* scores,
    int64_t scores_row_stride,
    int l_count,
    int s_count) {
  for (int i = 0; i < l_count; ++i) {
    row_max[i] =
        max_update_impl(row_max[i], scores + i * scores_row_stride, s_count);
  }
}

inline void scale_inplace_impl(float* buf, float scale, int64_t len) {
  int64_t i = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= len; i += 4) {
    float32x4_t v = vld1q_f32(buf + i);
    v = vmulq_f32(v, vs);
    vst1q_f32(buf + i, v);
  }
#endif
  for (; i < len; ++i) {
    buf[i] *= scale;
  }
}

inline float fp16_bits_to_float_impl(uint16_t h) {
  if ((h & 0x7fff) == 0) {
    return (h & 0x8000) ? -0.0f : 0.0f;
  }
  if (h == 0xbc00) {
    return -1.0f;
  }
  if (h == 0xfc00) {
    return -std::numeric_limits<float>::infinity();
  }
  if (h == 0x7c00) {
    return std::numeric_limits<float>::infinity();
  }

  const int sign = (h >> 15) & 0x1;
  const int exp = (h >> 10) & 0x1f;
  const int mant = h & 0x3ff;

  float value;
  if (exp == 0) {
    value = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
  } else if (exp == 0x1f) {
    value = mant == 0
        ? std::numeric_limits<float>::infinity()
        : std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, exp - 15);
  }
  return sign ? -value : value;
}

inline const char* mask_f16_row_ptr_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l,
    int64_t s) {
  const int64_t mb = b % p.mask_f16_ne3;
  const int64_t mh = n % p.mask_f16_ne2;
  return reinterpret_cast<const char*>(p.mask_f16_ptr)
      + s * p.mask_f16_nb0
      + l * p.mask_f16_nb1
      + mh * p.mask_f16_nb2
      + mb * p.mask_f16_nb3;
}

inline float load_mask_f16_from_ptr_impl(const char* src) {
  uint16_t bits = 0;
  std::memcpy(&bits, src, sizeof(bits));
  return fp16_bits_to_float_impl(bits);
}

inline float load_mask_f16_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l,
    int64_t s) {
  return load_mask_f16_from_ptr_impl(
      mask_f16_row_ptr_impl(p, b, n, l, s));
}

enum class MaskBlockKind {
  kMixed,
  kAllZero,
  kAllOff,
};

inline bool mask_f16_bits_is_off_impl(uint16_t bits) {
  if (bits == 0xfc00) {
    return true;
  }
  if ((bits & 0x8000) == 0) {
    return false;
  }
  if ((bits & 0x7c00) == 0x7c00) {
    return false;
  }
  // Negative finite F16 values are monotonic by raw bits. 0xe3d0 is
  // float16(-1000), so larger negative finite bit patterns are "off".
  return bits >= 0xe3d0;
}

inline MaskBlockKind classify_mask_f16_block_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l_start,
    int64_t s_start,
    int l_count,
    int s_count) {
  if (s_count == 8 && p.mask_f16_nb0 == static_cast<int64_t>(sizeof(uint16_t))) {
    constexpr uint64_t kF16AbsMask4 = 0x7fff7fff7fff7fffULL;
    constexpr uint64_t kF16NegInf4 = 0xfc00fc00fc00fc00ULL;
    bool all_zero_8 = true;
    bool all_neginf_8 = true;
    for (int i = 0; i < l_count; ++i) {
      const char* mask_row =
          mask_f16_row_ptr_impl(p, b, n, l_start + i, s_start);
      uint64_t lo = 0;
      uint64_t hi = 0;
      std::memcpy(&lo, mask_row, sizeof(lo));
      std::memcpy(&hi, mask_row + sizeof(lo), sizeof(hi));
      all_zero_8 = all_zero_8 && (((lo | hi) & kF16AbsMask4) == 0);
      all_neginf_8 = all_neginf_8 && lo == kF16NegInf4 && hi == kF16NegInf4;
      if (!all_zero_8 && !all_neginf_8) {
        break;
      }
    }
    if (all_zero_8) {
      return MaskBlockKind::kAllZero;
    }
    if (all_neginf_8) {
      return MaskBlockKind::kAllOff;
    }
  }

  bool all_zero = true;
  bool all_off = true;
  for (int i = 0; i < l_count; ++i) {
    const char* mask_row =
        mask_f16_row_ptr_impl(p, b, n, l_start + i, s_start);
    for (int j = 0; j < s_count; ++j) {
      uint16_t bits = 0;
      std::memcpy(&bits, mask_row + j * p.mask_f16_nb0, sizeof(bits));
      const bool is_zero = (bits & 0x7fff) == 0;
      all_zero = all_zero && is_zero;
      all_off = all_off && !is_zero && mask_f16_bits_is_off_impl(bits);
      if (!all_zero && !all_off) {
        return MaskBlockKind::kMixed;
      }
    }
  }
  if (all_zero) {
    return MaskBlockKind::kAllZero;
  }
  if (all_off) {
    return MaskBlockKind::kAllOff;
  }
  return MaskBlockKind::kMixed;
}

inline const char* mask_f32_row_ptr_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l,
    int64_t s) {
  const int64_t mb = b % p.mask_f32_ne3;
  const int64_t mh = n % p.mask_f32_ne2;
  return reinterpret_cast<const char*>(p.mask_f32_ptr)
      + s * p.mask_f32_nb0
      + l * p.mask_f32_nb1
      + mh * p.mask_f32_nb2
      + mb * p.mask_f32_nb3;
}

inline float load_mask_f32_from_ptr_impl(const char* src) {
  float value = 0.0f;
  std::memcpy(&value, src, sizeof(value));
  return value;
}

inline bool mask_f32_bits_is_off_impl(uint32_t bits) {
  if (bits == 0xff800000u) {
    return true;
  }
  // Negative finite fp32 values are monotonic by raw bits. 0xc47a0000 is
  // -1000.0f, so larger finite negative bit patterns are "off".
  return bits >= 0xc47a0000u && bits < 0xff800000u;
}

inline MaskBlockKind classify_mask_f32_block_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l_start,
    int64_t s_start,
    int l_count,
    int s_count) {
  if (s_count == 8 && p.mask_f32_nb0 == static_cast<int64_t>(sizeof(float))) {
    constexpr uint64_t kF32AbsMask2 = 0x7fffffff7fffffffULL;
    bool all_zero_8 = true;
    bool all_off_8 = true;
    for (int i = 0; i < l_count; ++i) {
      const char* mask_row =
          mask_f32_row_ptr_impl(p, b, n, l_start + i, s_start);
      for (int pair = 0; pair < 4; ++pair) {
        uint64_t bits64 = 0;
        std::memcpy(&bits64, mask_row + pair * sizeof(bits64),
                    sizeof(bits64));
        all_zero_8 = all_zero_8 && ((bits64 & kF32AbsMask2) == 0);
        const uint32_t lo = static_cast<uint32_t>(bits64);
        const uint32_t hi = static_cast<uint32_t>(bits64 >> 32);
        all_off_8 = all_off_8 &&
            mask_f32_bits_is_off_impl(lo) &&
            mask_f32_bits_is_off_impl(hi);
      }
      if (!all_zero_8 && !all_off_8) {
        break;
      }
    }
    if (all_zero_8) {
      return MaskBlockKind::kAllZero;
    }
    if (all_off_8) {
      return MaskBlockKind::kAllOff;
    }
  }

  bool all_zero = true;
  bool all_off = true;
  for (int i = 0; i < l_count; ++i) {
    const char* mask_row =
        mask_f32_row_ptr_impl(p, b, n, l_start + i, s_start);
    for (int j = 0; j < s_count; ++j) {
      uint32_t bits = 0;
      std::memcpy(&bits, mask_row + j * p.mask_f32_nb0, sizeof(bits));
      const bool is_zero = (bits & 0x7fffffffu) == 0;
      all_zero = all_zero && is_zero;
      all_off = all_off && mask_f32_bits_is_off_impl(bits);
      if (!all_zero && !all_off) {
        return MaskBlockKind::kMixed;
      }
    }
  }
  if (all_zero) {
    return MaskBlockKind::kAllZero;
  }
  if (all_off) {
    return MaskBlockKind::kAllOff;
  }
  return MaskBlockKind::kMixed;
}

inline void mark_pv_skip_block_impl(
    uint8_t* pv_skip_s,
    bool& pv_skip_initialized,
    bool& has_pv_skip,
    int64_t Sc_cur,
    int64_t s_start,
    int s_count) {
  if (!pv_skip_initialized) {
    std::memset(pv_skip_s, 0, static_cast<size_t>(Sc_cur));
    pv_skip_initialized = true;
  }
  for (int j = 0; j < s_count; ++j) {
    pv_skip_s[s_start + j] = 1;
  }
  has_pv_skip = true;
}

inline void fill_scores_block_impl(
    float* scores,
    int64_t scores_row_stride,
    int l_count,
    int s_count,
    float value) {
  for (int i = 0; i < l_count; ++i) {
    float* row = scores + i * scores_row_stride;
    for (int j = 0; j < s_count; ++j) {
      row[j] = value;
    }
  }
}

inline void add_mask_f16_to_scores_block_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l_start,
    int64_t s_start,
    int l_count,
    int s_count,
    float* scores,
    int64_t scores_row_stride) {
  for (int i = 0; i < l_count; ++i) {
    const char* mask_row =
        mask_f16_row_ptr_impl(p, b, n, l_start + i, s_start);
    float* row = scores + i * scores_row_stride;
    int j = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    if (p.mask_f16_nb0 == static_cast<int64_t>(sizeof(uint16_t))) {
      const uint16_t* m = reinterpret_cast<const uint16_t*>(mask_row);
      for (; j + 4 <= s_count; j += 4) {
        const float32x4_t mf = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(m + j)));
        vst1q_f32(row + j, vaddq_f32(vld1q_f32(row + j), mf));
      }
    }
#endif
    for (; j < s_count; ++j) {
      row[j] += load_mask_f16_from_ptr_impl(mask_row + j * p.mask_f16_nb0);
    }
  }
}

inline void add_mask_f32_to_scores_block_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l_start,
    int64_t s_start,
    int l_count,
    int s_count,
    float* scores,
    int64_t scores_row_stride) {
  for (int i = 0; i < l_count; ++i) {
    const char* mask_row =
        mask_f32_row_ptr_impl(p, b, n, l_start + i, s_start);
    float* row = scores + i * scores_row_stride;
    int j = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    if (p.mask_f32_nb0 == static_cast<int64_t>(sizeof(float))) {
      const float* m = reinterpret_cast<const float*>(mask_row);
      for (; j + 4 <= s_count; j += 4) {
        vst1q_f32(row + j, vaddq_f32(vld1q_f32(row + j), vld1q_f32(m + j)));
      }
    }
#endif
    for (; j < s_count; ++j) {
      row[j] += load_mask_f32_from_ptr_impl(mask_row + j * p.mask_f32_nb0);
    }
  }
}

inline bool has_direct_mask_impl(const SdpaParams& p) {
  return p.mask_f16_ptr != nullptr || p.mask_f32_ptr != nullptr;
}

inline MaskBlockKind classify_direct_mask_block_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l_start,
    int64_t s_start,
    int l_count,
    int s_count) {
  if (p.mask_f16_ptr != nullptr) {
    return classify_mask_f16_block_impl(
        p, b, n, l_start, s_start, l_count, s_count);
  }
  if (p.mask_f32_ptr != nullptr) {
    return classify_mask_f32_block_impl(
        p, b, n, l_start, s_start, l_count, s_count);
  }
  return MaskBlockKind::kMixed;
}

inline void add_direct_mask_to_scores_block_impl(
    const SdpaParams& p,
    int64_t b,
    int64_t n,
    int64_t l_start,
    int64_t s_start,
    int l_count,
    int s_count,
    float* scores,
    int64_t scores_row_stride) {
  if (p.mask_f16_ptr != nullptr) {
    add_mask_f16_to_scores_block_impl(
        p, b, n, l_start, s_start, l_count, s_count,
        scores, scores_row_stride);
  } else if (p.mask_f32_ptr != nullptr) {
    add_mask_f32_to_scores_block_impl(
        p, b, n, l_start, s_start, l_count, s_count,
        scores, scores_row_stride);
  }
}

// ──────────────────────────────────────────────────────────────────────
// process_q_tile_lc<MK, scalar_t, kPackedV, kHasMask, kCausal>
//
// kPackedV=false（默认）：原 l3kv 行为，V_tile = Vbase + k_off * v_stride_s + ev_off
// kPackedV=true ：V layout 是 [B, N, Ev/8, S, 8]，
//                 V_tile = Vbase + (ev_off>>3) * v_evblock_stride + k_off * 8
//                 喂给微内核的 v_row_stride 为字面量 8（编译期常量）。
//
// kHasMask / kCausal：编译期布尔，用 `if constexpr` 把 mask add 与 causal
// mask 块的整段代码（包括 causal_lim 的初始化、is_causal break 检查）从
// 不需要它们的实例化里彻底剔除。SDPA 入口按 4 种组合分发模板实例。
// ──────────────────────────────────────────────────────────────────────

template <class MK, typename scalar_t,
          bool kPackedV = false,
          bool kHasMask = false,
          bool kCausal  = false>
inline void process_q_tile_lc(
    const scalar_t* q_ptr,
    const scalar_t* k_ptr,
    const scalar_t* v_ptr,
    int64_t b, int64_t n, int64_t q0_outer,
    int64_t Lc_eff,
    const SdpaParams& p,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_l,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_s,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_s,
    int64_t v_evblock_stride,        // 仅 kPackedV=true 时有意义；非 packed 传 0
    int64_t m_stride_b, int64_t m_stride_n, int64_t m_stride_l,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_l,
    const TileSizes& ts,
    float* scores_l1,
    float* P_hat,
    float* O_acc,
    float* running_max,
    float* running_sum
    ) {
  const scalar_t* Qrow0 = q_ptr + b * q_stride_b + n * q_stride_n
                                + q0_outer * q_stride_l;

  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kInit);
    // 初始化 q_tile 状态。
    for (int64_t i = 0; i < Lc_eff; ++i) {
      running_max[i] = p.neg_inf;
      running_sum[i] = 0.0f;
    }
    std::memset(O_acc, 0, sizeof(float) * Lc_eff * p.Ev);
  }

  // causal_lim 仅在 kCausal=true 实例化里使用；非 causal 路径上保留数组
  // 占位（[[maybe_unused]]）以避免后面 step 3 引用时出现 undeclared，
  // 但**初始化循环放进 if constexpr**，省 8 次写入 + 一次比较链。
  [[maybe_unused]] int64_t causal_lim[64];
  [[maybe_unused]] int64_t max_causal = -1;
  if constexpr (kCausal) {
    for (int64_t i = 0; i < Lc_eff; ++i) {
      int64_t l_idx = q0_outer + i;
      causal_lim[i] = l_idx + p.causal_offset;
      if (causal_lim[i] > max_causal) max_causal = causal_lim[i];
    }
  }

  const int64_t LQ_INNER = 4;
  const int64_t num_inner = ceil_div_pos(Lc_eff, LQ_INNER);
  const int64_t Sc_l2 = ts.Sc_l2;
  (void)Sc_l2;

  for (int64_t s_l3 = 0; s_l3 < p.S; s_l3 += ts.Sc_l3) {
    const int64_t s_l3_end = std::min(s_l3 + ts.Sc_l3, p.S);

    if constexpr (kCausal) {
      if (s_l3 > max_causal) break;
    }

    for (int64_t s_l2 = s_l3; s_l2 < s_l3_end; s_l2 += ts.Sc_l2) {
      const int64_t s_l2_end = std::min(s_l2 + ts.Sc_l2, s_l3_end);
      const int64_t Sc_cur = s_l2_end - s_l2;

      if constexpr (kCausal) {
        if (s_l2 > max_causal) break;
      }

      // ── 双缓冲 L2 软件预取 ──
      // packed 与非 packed 都按 v_stride_s 跳一行：non-packed 下 v_stride_s = Ev,
      // packed 下 v_stride_s = 8。两者的 prefetch 起点都是「下一段 KV tile
      // 起始的若干 cache line」，逻辑等价。
      {
        const int64_t s_next = s_l2 + ts.Sc_l2;
        if (s_next < s_l3_end) {
          const scalar_t* K_base_next = k_ptr + b * k_stride_b
                                              + n * k_stride_n;
          const scalar_t* K_next =
              k_tile_ptr_impl<MK, scalar_t>(
                  K_base_next, s_next, k_stride_s, p.E);
          const scalar_t* V_next = v_ptr + b * v_stride_b
                                         + n * v_stride_n
                                         + s_next * v_stride_s;
          for (int line = 0; line < 4; ++line) {
            prefetch_l2_keep_impl(reinterpret_cast<const char*>(K_next) + line * 64);
            prefetch_l2_keep_impl(reinterpret_cast<const char*>(V_next) + line * 64);
          }
        }
      }

      // K_tile / V_tile 起始指针。
      const scalar_t* Kbase = k_ptr + b * k_stride_b + n * k_stride_n;
      const scalar_t* Krow0 =
          k_tile_ptr_impl<MK, scalar_t>(Kbase, s_l2, k_stride_s, p.E);
      // packed 路径下 Vbase 指向 [b][n][ev_block=0][s_l2][lane=0]，
      // 后面再按 (ev_off>>3) * v_evblock_stride 跳到正确 ev_block。
      const scalar_t* Vbase = v_ptr + b * v_stride_b + n * v_stride_n
                                    + s_l2 * v_stride_s;

      for (int line = 0; line < 2; ++line) {
        prefetch_l1_keep_impl(reinterpret_cast<const char*>(Krow0) + line * 64);
      }

      for (int64_t qi_inner = 0; qi_inner < num_inner; ++qi_inner) {
        const int64_t q0_inner = q0_outer + qi_inner * LQ_INNER;
        const int Lq_eff = static_cast<int>(
            std::min<int64_t>(LQ_INNER, Lc_eff - qi_inner * LQ_INNER));

        float* scores_8 = scores_l1 + qi_inner * LQ_INNER * Sc_cur;
        float* p_hat_8  = P_hat     + qi_inner * LQ_INNER * Sc_cur;
        float* o_acc_8  = O_acc     + qi_inner * LQ_INNER * p.Ev;
        float* rmax_8   = running_max + qi_inner * LQ_INNER;
        float* rsum_8   = running_sum + qi_inner * LQ_INNER;

        const scalar_t* Qrow_inner = Qrow0 + qi_inner * LQ_INNER * q_stride_l;
        int64_t Sc_active = Sc_cur;
        int64_t softmax_len[4];
        for (int i = 0; i < 4; ++i) softmax_len[i] = Sc_cur;
        if constexpr (kCausal) {
          const int64_t q_inner_max_causal =
              causal_lim[qi_inner * LQ_INNER + Lq_eff - 1];
          Sc_active = std::min<int64_t>(
              Sc_cur, q_inner_max_causal - s_l2 + 1);
          if (Sc_active <= 0) {
            continue;
          }
          for (int i = 0; i < Lq_eff; ++i) {
            const int64_t lim = causal_lim[qi_inner * LQ_INNER + i];
            softmax_len[i] = std::max<int64_t>(
                0, std::min<int64_t>(Sc_active, lim - s_l2 + 1));
          }
          for (int i = Lq_eff; i < 4; ++i) {
            softmax_len[i] = 0;
          }
        }

        if (qi_inner + 2 < num_inner) {
          for (int line = 0; line < 2; ++line) {
            prefetch_l1_keep_impl(reinterpret_cast<const char*>(Vbase) + line * 64);
          }
        }

        constexpr int64_t kMaxTrackedPvSkip = 4096;
        uint8_t pv_skip_s[kMaxTrackedPvSkip];
        const bool has_direct_mask =
            kHasMask && has_direct_mask_impl(p);
        const bool track_pv_skip =
            has_direct_mask && Sc_cur <= kMaxTrackedPvSkip;
        bool pv_skip_initialized = false;
        bool has_pv_skip = false;

        {
          FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kQkt);
          // ── 步骤 1: scores[Lq_eff][Sc_cur] = scale * Q · K^T ──
          int64_t s_off = 0;
          for (; s_off + 4 <= Sc_active; s_off += 4) {
            const scalar_t* K_tile =
                k_tile_ptr_impl<MK, scalar_t>(
                    Kbase, s_l2 + s_off, k_stride_s, p.E);
            MaskBlockKind mask_kind = MaskBlockKind::kMixed;
            if constexpr (kHasMask) {
              if (has_direct_mask) {
                mask_kind = classify_direct_mask_block_impl(
                    p, b, n, q0_inner, s_l2 + s_off, Lq_eff, 4);
              }
            }
            if (mask_kind == MaskBlockKind::kAllOff) {
              fill_scores_block_impl(
                  scores_8 + s_off, Sc_cur, Lq_eff, 4, p.neg_inf);
              if (track_pv_skip) {
                mark_pv_skip_block_impl(
                    pv_skip_s, pv_skip_initialized, has_pv_skip,
                    Sc_cur, s_off, 4);
              }
            } else if (Lq_eff == 4) {
              MK::qkt_4x4(Qrow_inner, q_stride_l, K_tile, k_stride_s,
                          p.E, p.scale_f,
                          scores_8 + s_off, Sc_cur);
            } else {
              MK::qkt_4x4_tail(Qrow_inner, q_stride_l, K_tile, k_stride_s,
                               p.E, p.scale_f,
                               scores_8 + s_off, Sc_cur,
                               Lq_eff, 4);
            }
            if constexpr (kHasMask) {
              if (has_direct_mask && mask_kind == MaskBlockKind::kMixed) {
                add_direct_mask_to_scores_block_impl(
                    p, b, n, q0_inner, s_l2 + s_off, Lq_eff, 4,
                    scores_8 + s_off, Sc_cur);
              }
            }
          }
          if (s_off < Sc_active) {
            const scalar_t* K_tile =
                k_tile_ptr_impl<MK, scalar_t>(
                    Kbase, s_l2 + s_off, k_stride_s, p.E);
            const int tail_width = static_cast<int>(Sc_active - s_off);
            MaskBlockKind mask_kind = MaskBlockKind::kMixed;
            if constexpr (kHasMask) {
              if (has_direct_mask) {
                mask_kind = classify_direct_mask_block_impl(
                    p, b, n, q0_inner, s_l2 + s_off, Lq_eff, tail_width);
              }
            }
            if (mask_kind == MaskBlockKind::kAllOff) {
              fill_scores_block_impl(
                  scores_8 + s_off, Sc_cur, Lq_eff, tail_width, p.neg_inf);
              if (track_pv_skip) {
                mark_pv_skip_block_impl(
                    pv_skip_s, pv_skip_initialized, has_pv_skip,
                    Sc_cur, s_off, tail_width);
              }
            } else {
              MK::qkt_4x4_tail(Qrow_inner, q_stride_l, K_tile, k_stride_s,
                               p.E, p.scale_f,
                               scores_8 + s_off, Sc_cur,
                               Lq_eff, tail_width);
            }
            if constexpr (kHasMask) {
              if (has_direct_mask && mask_kind == MaskBlockKind::kMixed) {
                add_direct_mask_to_scores_block_impl(
                    p, b, n, q0_inner, s_l2 + s_off, Lq_eff, tail_width,
                    scores_8 + s_off, Sc_cur);
              }
            }
            s_off = Sc_active;
          }
        }

        // ── 步骤 2: additive attention mask（kHasMask 编译期开关）──
        // mask 块每个 8x Lq_eff 行子块都要全量加 mask，原标量循环 IPC 受
        // load/store 串行约束（A-class 一周 2 LSU）。改 vld/vadd/vst：
        // fp32 4-wide 主体 + 标量尾，主体 IPC ~3，尾巴 ≤ 3 次。
        if constexpr (kHasMask) {
          FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kMask);
          if (has_direct_mask) {
            // Direct ggml mask was applied while writing QK blocks to scores_8.
          } else {
            for (int i = 0; i < Lq_eff; ++i) {
              const float* m_row = p.mask_ptr + b * m_stride_b
                                              + n * m_stride_n
                                              + (q0_inner + i) * m_stride_l
                                              + s_l2;
              float* sc_row = scores_8 + i * Sc_cur;
              int64_t j = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
              for (; j + 4 <= Sc_active; j += 4) {
                const float32x4_t v =
                    vaddq_f32(vld1q_f32(sc_row + j), vld1q_f32(m_row + j));
                vst1q_f32(sc_row + j, v);
              }
#endif
              for (; j < Sc_active; ++j) {
                sc_row[j] += m_row[j];
              }
            }
          }
        }

        // ── 步骤 3: causal mask（按 row；kCausal 编译期开关）──
        if constexpr (kCausal) {
          // Row-specific causal visibility is represented by softmax_len.
          // QK only computed the prefix shared by at least one row in this
          // 4-row Q block; invalid row tails are zeroed in P before PV.
        }

        float new_max[4];
        {
          FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kSoftmax);
          // ── 步骤 4–7: 逐行 online softmax（plain：扫 scores 求 row max）──
          float row_max[4];
          for (int i = 0; i < 4; ++i) row_max[i] = p.neg_inf;
          for (int i = 0; i < Lq_eff; ++i) {
            const float* sc_row = scores_8 + i * Sc_cur;
            float m = p.neg_inf;
            const int64_t len_i = softmax_len[i];
            int64_t j = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
            float32x4_t vm = vdupq_n_f32(p.neg_inf);
            float32x4_t vm1 = vdupq_n_f32(p.neg_inf);
            for (; j + 8 <= len_i; j += 8) {
              vm = vmaxq_f32(vm, vld1q_f32(sc_row + j));
              vm1 = vmaxq_f32(vm1, vld1q_f32(sc_row + j + 4));
            }
            for (; j + 4 <= len_i; j += 4) {
              vm = vmaxq_f32(vm, vld1q_f32(sc_row + j));
            }
            m = vmaxvq_f32(vmaxq_f32(vm, vm1));
#endif
            for (; j < len_i; ++j) {
              if (sc_row[j] > m) m = sc_row[j];
            }
            row_max[i] = m;
          }

          float correction[4];
          for (int i = 0; i < Lq_eff; ++i) {
            if (rmax_8[i] == p.neg_inf) {
              new_max[i] = row_max[i];
              correction[i] = 1.0f;
            } else {
              new_max[i] = std::max(rmax_8[i], row_max[i]);
              correction[i] = std::exp(rmax_8[i] - new_max[i]);
            }
            rsum_8[i] *= correction[i];
            if (correction[i] != 1.0f) {
              scale_inplace_impl(o_acc_8 + i * p.Ev, correction[i], p.Ev);
            }
          }

          for (int i = 0; i < Lq_eff; ++i) {
            const float* sc_row = scores_8 + i * Sc_cur;
            float* p_row = p_hat_8 + i * Sc_cur;
            const int64_t len_i = softmax_len[i];
            float row_sum = 0.0f;
            if (new_max[i] == p.neg_inf) {
              std::memset(p_row, 0, sizeof(float) * Sc_cur);
            } else {
              row_sum = vectorized_exp_minus_impl(
                  p_row, sc_row, new_max[i], len_i);
            }
            if (len_i < Sc_cur) {
              std::memset(p_row + len_i, 0,
                          sizeof(float) * (Sc_cur - len_i));
            }
            rsum_8[i] += row_sum;
          }
          for (int i = Lq_eff; i < 4; ++i) {
            std::memset(p_hat_8 + i * Sc_cur, 0, sizeof(float) * Sc_cur);
          }
        }

        // ── 步骤 8: O_acc += P_hat · V ──
        // V_tile / v_row_stride 走两条编译期分支：
        //   kPackedV=false → V[b, n, s_l2+k_off, ev_off:ev_off+8]，stride=Ev
        //   kPackedV=true  → V_packed[b, n, ev_off/8, s_l2+k_off, 0:8]，stride=8
        //
        // 性能历史：早期实现把外层切成 `for (k_off += 8) pv_8x8(..., Sk=8)`
        // 共 Sc_cur/8 次调用——但 PV microkernel 内部已有完整 k 循环（bf16
        // 标量 + fp32/pquad 4-way 软件流水 + 标量尾），完全能吞 Sk=Sc_cur。
        // 拆成多次调用的唯一后果是每次都付 16 vld_O + 16 vst_O 的边界
        // round-trip（约 5–10 cycle 串行损失）。合并为单次调用后，每个 ev_off
        // 块只付一次 O 累加器加载/写回，PV 内部软件流水跨整个 Sc_cur 不打断。
        // 数值上完全等价：fma 累加序按 k 单调递增，未变。
        {
          FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kPv);
          constexpr int64_t kPvEvStep = 4;
          for (int64_t ev_off = 0; ev_off < p.Ev; ev_off += kPvEvStep) {
            // packv layout 下 ev_off += 4 跳到下一个 ev_block（物理地址跨
            // S*16 字节），远超 stride prefetcher 跟踪窗口；切块时把下一个
            // ev_block 起始 2 cache line 提前进 L1，与 PV inner 循环并行。
            if constexpr (kPackedV) {
              if (ev_off + kPvEvStep < p.Ev) {
                const scalar_t* next_block =
                    Vbase + ((ev_off + kPvEvStep) >> 2) * v_evblock_stride;
                for (int line = 0; line < 2; ++line) {
                  prefetch_l1_keep_impl(
                      reinterpret_cast<const char*>(next_block) + line * 64);
                }
              }
            }

            const int64_t Ev_cur = std::min<int64_t>(kPvEvStep, p.Ev - ev_off);

            const scalar_t* V_tile;
            int64_t v_row_stride_for_mk;
            if constexpr (kPackedV) {
              V_tile = Vbase + (ev_off >> 2) * v_evblock_stride;  // k=0 起点
              v_row_stride_for_mk = 4;
            } else {
              V_tile = Vbase + ev_off;                            // k=0 起点
              v_row_stride_for_mk = v_stride_s;
            }
            float* O_tile = o_acc_8 + ev_off;

            auto run_pv_segment = [&](int64_t k_start, int64_t k_len) {
              if (k_len <= 0) {
                return;
              }
              const scalar_t* V_seg = V_tile + k_start * v_row_stride_for_mk;
              if (Lq_eff == 4 && Ev_cur == 4) {
                MK::pv_4x4(p_hat_8 + k_start, Sc_cur,
                           V_seg, v_row_stride_for_mk,
                           k_len,
                           O_tile, p.Ev);
              } else {
                MK::pv_4x4_tail(p_hat_8 + k_start, Sc_cur,
                                V_seg, v_row_stride_for_mk,
                                k_len,
                                O_tile, p.Ev,
                                Lq_eff, static_cast<int>(Ev_cur));
              }
            };

            if (has_pv_skip) {
              int64_t k = 0;
              while (k < Sc_active) {
                while (k < Sc_active && pv_skip_s[k] != 0) {
                  ++k;
                }
                const int64_t k_start = k;
                while (k < Sc_active && pv_skip_s[k] == 0) {
                  ++k;
                }
                run_pv_segment(k_start, k - k_start);
              }
            } else {
              run_pv_segment(0, Sc_active);
            }
          }
        }

        // ── 步骤 9: running_max <- new_max ──
        for (int i = 0; i < Lq_eff; ++i) {
          rmax_8[i] = new_max[i];
        }
      }  // qi_inner
    }    // s_l2
  }      // s_l3

  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kFinalize);
    // ── 归一化并写入 o_row ──
    for (int64_t i = 0; i < Lc_eff; ++i) {
      float* o_row = p.out_ptr + b * o_stride_b + n * o_stride_n
                               + (q0_outer + i) * o_stride_l;
      if (p.max_logits_ptr != nullptr || p.lse_ptr != nullptr) {
        const int64_t stats_idx = b * (p.N * p.L) + n * p.L + q0_outer + i;
        if (p.max_logits_ptr != nullptr) {
          p.max_logits_ptr[stats_idx] = running_max[i];
        }
        if (p.lse_ptr != nullptr) {
          p.lse_ptr[stats_idx] =
              running_sum[i] > 0.0f
                  ? running_max[i] + std::log(running_sum[i])
                  : std::numeric_limits<float>::infinity();
        }
      }
      if (running_sum[i] > 0.0f) {
        const float inv_sum = 1.0f / running_sum[i];
        int64_t ev = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
        const float32x4_t vinv = vdupq_n_f32(inv_sum);
        for (; ev + 4 <= p.Ev; ev += 4) {
          float32x4_t v = vld1q_f32(O_acc + i * p.Ev + ev);
          vst1q_f32(o_row + ev, vmulq_f32(v, vinv));
        }
#endif
        for (; ev < p.Ev; ++ev) {
          o_row[ev] = O_acc[i * p.Ev + ev] * inv_sum;
        }
      } else {
        for (int64_t ev = 0; ev < p.Ev; ++ev) {
          o_row[ev] = 0.0f;
        }
      }
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// 路径 A：collapse(3) over (B, N, q_tile)
// ──────────────────────────────────────────────────────────────────────

template <class MK, typename scalar_t,
          bool kPackedV = false,
          bool kHasMask = false,
          bool kCausal  = false>
inline void run_path_collapse3(
    const scalar_t* q_ptr,
    const scalar_t* k_ptr,
    const scalar_t* v_ptr,
    const SdpaParams& p,
    const TileSizes& ts,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_l,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_s,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_s,
    int64_t v_evblock_stride,
    int64_t m_stride_b, int64_t m_stride_n, int64_t m_stride_l,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_l) {
  const int64_t LQ_OUTER = ts.Lc_l2;
  const int64_t num_q_tiles = ceil_div_pos(p.L, LQ_OUTER);
  const int64_t Sc_l2 = ts.Sc_l2;

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    const int64_t sc_max = std::max<int64_t>(8, Sc_l2);
    std::vector<float> scores_l1_vec(LQ_OUTER * sc_max);
    std::vector<float> P_hat_vec(LQ_OUTER * sc_max);
    std::vector<float> O_acc_vec(LQ_OUTER * p.Ev);
    std::vector<float> running_max_vec(LQ_OUTER);
    std::vector<float> running_sum_vec(LQ_OUTER);

#ifdef _OPENMP
    #pragma omp for collapse(3) schedule(static)
#endif
    for (int64_t b = 0; b < p.B; ++b) {
      for (int64_t n = 0; n < p.N; ++n) {
        for (int64_t qi_outer = 0; qi_outer < num_q_tiles; ++qi_outer) {
          const int64_t q0_outer = qi_outer * LQ_OUTER;
          const int64_t Lc_eff = std::min<int64_t>(LQ_OUTER, p.L - q0_outer);
          process_q_tile_lc<MK, scalar_t, kPackedV, kHasMask, kCausal>(
              q_ptr, k_ptr, v_ptr,
              b, n, q0_outer, Lc_eff, p,
              q_stride_b, q_stride_n, q_stride_l,
              k_stride_b, k_stride_n, k_stride_s,
              v_stride_b, v_stride_n, v_stride_s,
              v_evblock_stride,
              m_stride_b, m_stride_n, m_stride_l,
              o_stride_b, o_stride_n, o_stride_l,
              ts,
              scores_l1_vec.data(), P_hat_vec.data(),
              O_acc_vec.data(),
              running_max_vec.data(), running_sum_vec.data());
        }
      }
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// 路径 B：omp single + taskloop
// ──────────────────────────────────────────────────────────────────────

template <class MK, typename scalar_t,
          bool kPackedV = false,
          bool kHasMask = false,
          bool kCausal  = false>
inline void run_path_taskloop(
    const scalar_t* q_ptr,
    const scalar_t* k_ptr,
    const scalar_t* v_ptr,
    const SdpaParams& p,
    const TileSizes& ts,
    int num_groups,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_l,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_s,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_s,
    int64_t v_evblock_stride,
    int64_t m_stride_b, int64_t m_stride_n, int64_t m_stride_l,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_l) {
  const int64_t LQ_OUTER = ts.Lc_l2;
  const int64_t num_q_tiles = ceil_div_pos(p.L, LQ_OUTER);
  const int64_t Sc_l2 = ts.Sc_l2;
  const int64_t sc_max = std::max<int64_t>(8, Sc_l2);
  const int total_threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif

  std::vector<std::vector<float>> scores_l1_pool(total_threads);
  std::vector<std::vector<float>> p_hat_pool(total_threads);
  std::vector<std::vector<float>> o_acc_pool(total_threads);
  std::vector<std::vector<float>> rmax_pool(total_threads);
  std::vector<std::vector<float>> rsum_pool(total_threads);

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
    scores_l1_pool[tid].assign(LQ_OUTER * sc_max, 0.0f);
    p_hat_pool[tid].assign(LQ_OUTER * sc_max, 0.0f);
    o_acc_pool[tid].assign(LQ_OUTER * p.Ev, 0.0f);
    rmax_pool[tid].assign(LQ_OUTER, 0.0f);
    rsum_pool[tid].assign(LQ_OUTER, 0.0f);

#ifdef _OPENMP
    #pragma omp barrier
    #pragma omp single
#endif
    {
      const int64_t grain = std::max<int64_t>(
          1, num_q_tiles / std::max(1, num_groups));
      for (int64_t b = 0; b < p.B; ++b) {
        for (int64_t n = 0; n < p.N; ++n) {
#ifdef _OPENMP
          #pragma omp taskloop grainsize(grain) nogroup default(shared) \
              firstprivate(b, n)
#endif
          for (int64_t qi_outer = 0; qi_outer < num_q_tiles; ++qi_outer) {
            const int my_tid =
#ifdef _OPENMP
                omp_get_thread_num();
#else
                0;
#endif
            const int64_t q0_outer = qi_outer * LQ_OUTER;
            const int64_t Lc_eff = std::min<int64_t>(LQ_OUTER, p.L - q0_outer);
            process_q_tile_lc<MK, scalar_t, kPackedV, kHasMask, kCausal>(
                q_ptr, k_ptr, v_ptr,
                b, n, q0_outer, Lc_eff, p,
                q_stride_b, q_stride_n, q_stride_l,
                k_stride_b, k_stride_n, k_stride_s,
                v_stride_b, v_stride_n, v_stride_s,
                v_evblock_stride,
                m_stride_b, m_stride_n, m_stride_l,
                o_stride_b, o_stride_n, o_stride_l,
                ts,
                scores_l1_pool[my_tid].data(),
                p_hat_pool[my_tid].data(),
                o_acc_pool[my_tid].data(),
                rmax_pool[my_tid].data(),
                rsum_pool[my_tid].data());
          }
        }
      }
    }
  }
}


}  // namespace fused_cpp::sdpa_flash2_neon_l3kv_impl
