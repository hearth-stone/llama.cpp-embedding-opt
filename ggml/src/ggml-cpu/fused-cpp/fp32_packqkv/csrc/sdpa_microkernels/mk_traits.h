#pragma once
// Minimal micro-kernel trait helpers for the extracted fp32 packqkv target.
// The repository-level registry, benchmark, and framework dispatch code is
// intentionally omitted from this standalone copy.

#include <type_traits>

namespace fused_cpp::sdpa_microkernels {

constexpr int kMicroLq = 8;
constexpr int kMicroSk = 8;
constexpr int kMicroEv = 8;

namespace detail {

template <class MK, class = void>
struct has_enabled : std::false_type {};

template <class MK>
struct has_enabled<MK, std::void_t<decltype(MK::kEnabled)>>
    : std::bool_constant<MK::kEnabled> {};

}  // namespace detail

template <class MK>
inline constexpr bool mk_is_enabled_v = detail::has_enabled<MK>::value;

}  // namespace fused_cpp::sdpa_microkernels
