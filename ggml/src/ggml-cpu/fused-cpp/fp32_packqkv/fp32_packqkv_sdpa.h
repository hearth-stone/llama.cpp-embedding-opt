#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace fp32_packqkv_sdpa {

template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    void* ptr = nullptr;
    if (n != 0 && ::posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t) noexcept {
    ::free(ptr);
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
inline bool operator==(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
inline bool operator!=(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return false;
}

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

struct Config {
  int64_t B = 1;
  int64_t N = 8;
  int64_t L = 512;
  int64_t S = 512;
  int64_t E = 64;
  int64_t Ev = 64;
  bool causal = false;
  float scale = 0.0f;
  int64_t causal_offset = 0;
  int64_t s_tile = 0;
};

void sdpa_fp32_packqkv_pbf16pv(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const Config& cfg);

void reference_sdpa_fp32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const Config& cfg);

void reference_sdpa_fp32_mask(
    const float* q,
    const float* k,
    const float* v,
    const float* mask32,
    float* out,
    const Config& cfg);

double counted_gflops(const Config& cfg, double mean_ms);
double checksum(const float* data, int64_t size);
double max_abs_diff(const float* a, const float* b, int64_t size);

}  // namespace fp32_packqkv_sdpa

#if defined(GGML_FUSED_CPP_BUILTIN) && (defined(__GNUC__) || defined(__clang__))
#define FUSED_CPP_FP32_PACKQKV_API __attribute__((visibility("hidden")))
#elif defined(GGML_FUSED_CPP_BUILTIN)
#define FUSED_CPP_FP32_PACKQKV_API
#elif defined(_WIN32)
#define FUSED_CPP_FP32_PACKQKV_API __declspec(dllexport)
#else
#define FUSED_CPP_FP32_PACKQKV_API __attribute__((visibility("default")))
#endif

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
    float scale);

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
    float scale);

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
    float scale);

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
    float scale);
