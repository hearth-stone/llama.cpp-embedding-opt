#pragma once
// Shared SDPA packing helpers.
//
// The SVE path only changes the load/store primitive used by the packers. It
// keeps the exact byte layout of the existing memcpy implementations.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "sdpa_microkernels/neon_cache_config.h"

#if FUSED_CPP_SDPA_CACHE_HAS_SVE
#include <arm_sve.h>
#endif

namespace fused_cpp::sdpa_pack_utils {

inline void copy_bytes(const void* src, void* dst, size_t nbytes) {
#if FUSED_CPP_SDPA_CACHE_HAS_SVE
  const auto* s = static_cast<const uint8_t*>(src);
  auto* d = static_cast<uint8_t*>(dst);
  uint64_t j = 0;
  const uint64_t n = static_cast<uint64_t>(nbytes);
  while (j < n) {
    const svbool_t pg = svwhilelt_b8(j, n);
    const svuint8_t v = svld1_u8(pg, s + j);
    svst1_u8(pg, d + j, v);
    j += static_cast<uint64_t>(svcntb());
  }
#else
  std::memcpy(dst, src, nbytes);
#endif
}

inline void copy_u16x4(const uint16_t* src, uint16_t* dst) {
#if FUSED_CPP_SDPA_CACHE_HAS_SVE
  const svbool_t pg = svwhilelt_b16(uint64_t{0}, uint64_t{4});
  const svuint16_t v = svld1_u16(pg, src);
  svst1_u16(pg, dst, v);
#else
  std::memcpy(dst, src, 4 * sizeof(uint16_t));
#endif
}

inline void copy_f32x8(const float* src, float* dst) {
#if FUSED_CPP_SDPA_CACHE_HAS_SVE
  uint64_t j = 0;
  while (j < 8) {
    const svbool_t pg = svwhilelt_b32(j, uint64_t{8});
    const svfloat32_t v = svld1_f32(pg, src + j);
    svst1_f32(pg, dst + j, v);
    j += static_cast<uint64_t>(svcntw());
  }
#elif FUSED_CPP_SDPA_CACHE_HAS_NEON
  vst1q_f32(dst + 0, vld1q_f32(src + 0));
  vst1q_f32(dst + 4, vld1q_f32(src + 4));
#else
  std::memcpy(dst, src, 8 * sizeof(float));
#endif
}

template <typename scalar_t>
inline void copy_8_elems(const scalar_t* src, scalar_t* dst) {
#if FUSED_CPP_SDPA_CACHE_HAS_SVE
  if constexpr (sizeof(scalar_t) == sizeof(uint16_t)) {
    const auto* src_u16 = reinterpret_cast<const uint16_t*>(src);
    auto* dst_u16 = reinterpret_cast<uint16_t*>(dst);
    const svbool_t pg = svwhilelt_b16(uint64_t{0}, uint64_t{8});
    const svuint16_t v = svld1_u16(pg, src_u16);
    svst1_u16(pg, dst_u16, v);
  } else if constexpr (std::is_same_v<scalar_t, float>) {
    copy_f32x8(src, dst);
  } else {
    std::memcpy(dst, src, 8 * sizeof(scalar_t));
  }
#else
  std::memcpy(dst, src, 8 * sizeof(scalar_t));
#endif
}

template <typename scalar_t>
void pack_v_to_evblock8(
    const scalar_t* v_src,
    scalar_t* v_dst,
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t Ev) {
  const int64_t Eb = Ev / 8;
  const int64_t bn_stride_src = N * S * Ev;
  const int64_t n_stride_src = S * Ev;
  const int64_t bn_stride_dst = N * Eb * S * 8;
  const int64_t n_stride_dst = Eb * S * 8;
  const int64_t evblock_stride_dst = S * 8;

#ifdef _OPENMP
  #pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t eb = 0; eb < Eb; ++eb) {
        const scalar_t* src = v_src + b * bn_stride_src
                                    + n * n_stride_src
                                    + eb * 8;
        scalar_t* dst = v_dst + b * bn_stride_dst
                              + n * n_stride_dst
                              + eb * evblock_stride_dst;
        int64_t s = 0;
        for (; s + 4 <= S; s += 4) {
          copy_8_elems(src + (s + 0) * Ev, dst + (s + 0) * 8);
          copy_8_elems(src + (s + 1) * Ev, dst + (s + 1) * 8);
          copy_8_elems(src + (s + 2) * Ev, dst + (s + 2) * 8);
          copy_8_elems(src + (s + 3) * Ev, dst + (s + 3) * 8);
        }
        for (; s < S; ++s) {
          copy_8_elems(src + s * Ev, dst + s * 8);
        }
      }
    }
  }
}

}  // namespace fused_cpp::sdpa_pack_utils
