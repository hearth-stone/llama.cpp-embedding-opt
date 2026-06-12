#pragma once

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fp32_packqkv_standalone_shim {

inline void append_to_stream(std::ostringstream&) {}

template <typename T, typename... Rest>
inline void append_to_stream(std::ostringstream& os, T&& value, Rest&&... rest) {
  os << std::forward<T>(value);
  append_to_stream(os, std::forward<Rest>(rest)...);
}

template <typename... Args>
[[noreturn]] inline void throw_check(Args&&... args) {
  std::ostringstream os;
  append_to_stream(os, std::forward<Args>(args)...);
  throw std::runtime_error(os.str());
}

inline uint16_t float_to_bf16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_bits_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

}  // namespace fp32_packqkv_standalone_shim

namespace at {

struct BFloat16 {
  uint16_t x;

  BFloat16() : x(0) {}
  explicit BFloat16(float value)
      : x(fp32_packqkv_standalone_shim::float_to_bf16_bits(value)) {}

  BFloat16& operator=(float value) {
    x = fp32_packqkv_standalone_shim::float_to_bf16_bits(value);
    return *this;
  }

  operator float() const {
    return fp32_packqkv_standalone_shim::bf16_bits_to_float(x);
  }
};

static_assert(sizeof(BFloat16) == sizeof(uint16_t));
static_assert(alignof(BFloat16) == alignof(uint16_t));

}  // namespace at

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...)                                            \
  do {                                                                    \
    if (!(cond)) {                                                        \
      ::fp32_packqkv_standalone_shim::throw_check(__VA_ARGS__);           \
    }                                                                     \
  } while (false)
#endif
