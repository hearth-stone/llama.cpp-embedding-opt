#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sdpa_common.h"

namespace fused_cpp::sdpa_profile {

enum class Slot : int {
  kTotal = 0,
  kVAlloc,
  kVPack,
  kKAlloc,
  kKPack,
  kMain,
  kInit,
  kQPack,
  kQkt,
  kMask,
  kSoftmax,
  kPConvert,
  kPv,
  kFinalize,
  kKColPack,
  kQktMicro,
  kCount,
};

struct State {
  uint64_t ns[static_cast<int>(Slot::kCount)] = {};
  uint64_t calls[static_cast<int>(Slot::kCount)] = {};
};

inline State& state() {
  static State s;
  return s;
}

inline bool enabled() {
  static const bool on = []() {
    const char* env = std::getenv("FUSED_CPP_SDPA_PROFILE");
    return env != nullptr && std::strcmp(env, "0") != 0;
  }();
  return on;
}

inline bool deep_enabled() {
  static const bool on = []() {
    const char* env = std::getenv("FUSED_CPP_SDPA_PROFILE_DEEP");
    return enabled() && env != nullptr && std::strcmp(env, "0") != 0;
  }();
  return on;
}

inline uint64_t now_ns() {
  const auto t = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t).count());
}

inline void reset() {
  if (!enabled()) return;
  state() = State{};
}

inline void add(Slot slot, uint64_t ns) {
  if (!enabled()) return;
  const int i = static_cast<int>(slot);
  state().ns[i] += ns;
  state().calls[i] += 1;
}

class ScopedTimer {
 public:
  explicit ScopedTimer(Slot slot)
      : slot_(slot), start_(enabled() ? now_ns() : 0), active_(enabled()) {}

  ScopedTimer(Slot slot, bool active)
      : slot_(slot), start_(active ? now_ns() : 0), active_(active) {}

  ~ScopedTimer() {
    if (active_) {
      add(slot_, now_ns() - start_);
    }
  }

 private:
  Slot slot_;
  uint64_t start_;
  bool active_;
};

inline const char* slot_name(Slot slot) {
  switch (slot) {
    case Slot::kTotal: return "total";
    case Slot::kVAlloc: return "v_alloc";
    case Slot::kVPack: return "v_pack";
    case Slot::kKAlloc: return "k_alloc";
    case Slot::kKPack: return "k_pack";
    case Slot::kMain: return "main_compute";
    case Slot::kInit: return "tile_init";
    case Slot::kQPack: return "q_pack";
    case Slot::kQkt: return "qkt_total";
    case Slot::kMask: return "mask";
    case Slot::kSoftmax: return "softmax";
    case Slot::kPConvert: return "p_bf16_convert";
    case Slot::kPv: return "pv";
    case Slot::kFinalize: return "final_write";
    case Slot::kKColPack: return "k_col_pack";
    case Slot::kQktMicro: return "qkt_micro";
    case Slot::kCount: return "count";
  }
  return "unknown";
}

inline const char* dtype_name(SdpaDtype dtype) {
  switch (dtype) {
    case SdpaDtype::kFloat32: return "fp32";
    case SdpaDtype::kBFloat16: return "bf16";
  }
  return "unknown";
}

inline double ns_to_ms(uint64_t ns) {
  return static_cast<double>(ns) / 1.0e6;
}

inline void print_slot(Slot slot, uint64_t total_ns) {
  const int i = static_cast<int>(slot);
  const uint64_t ns = state().ns[i];
  if (ns == 0) return;
  const double pct = total_ns == 0
      ? 0.0
      : 100.0 * static_cast<double>(ns) / static_cast<double>(total_ns);
  std::fprintf(stderr,
               "[sdpa_profile]   %-16s %9.3f ms %6.2f%% calls=%llu\n",
               slot_name(slot),
               ns_to_ms(ns),
               pct,
               static_cast<unsigned long long>(state().calls[i]));
}

inline void print_summary(const char* version,
                          const char* kernel,
                          const SdpaParams& p,
                          const char* path) {
  if (!enabled()) return;
  const uint64_t total_ns = state().ns[static_cast<int>(Slot::kTotal)];
  std::fprintf(stderr,
               "[sdpa_profile] version=%s kernel=%s path=%s "
               "shape=B%lld-N%lld-L%lld-S%lld-E%lld-Ev%lld dtype=%s "
               "causal=%d total_ms=%.3f\n",
               version,
               kernel,
               path,
               static_cast<long long>(p.B),
               static_cast<long long>(p.N),
               static_cast<long long>(p.L),
               static_cast<long long>(p.S),
               static_cast<long long>(p.E),
               static_cast<long long>(p.Ev),
               dtype_name(p.dtype),
               p.is_causal ? 1 : 0,
               ns_to_ms(total_ns));

  print_slot(Slot::kVAlloc, total_ns);
  print_slot(Slot::kVPack, total_ns);
  print_slot(Slot::kKAlloc, total_ns);
  print_slot(Slot::kKPack, total_ns);
  print_slot(Slot::kMain, total_ns);
  print_slot(Slot::kInit, total_ns);
  print_slot(Slot::kQPack, total_ns);
  print_slot(Slot::kQkt, total_ns);
  print_slot(Slot::kKColPack, total_ns);
  print_slot(Slot::kQktMicro, total_ns);
  print_slot(Slot::kMask, total_ns);
  print_slot(Slot::kSoftmax, total_ns);
  print_slot(Slot::kPConvert, total_ns);
  print_slot(Slot::kPv, total_ns);
  print_slot(Slot::kFinalize, total_ns);

  const uint64_t top_ns =
      state().ns[static_cast<int>(Slot::kVAlloc)] +
      state().ns[static_cast<int>(Slot::kVPack)] +
      state().ns[static_cast<int>(Slot::kKAlloc)] +
      state().ns[static_cast<int>(Slot::kKPack)] +
      state().ns[static_cast<int>(Slot::kMain)];
  if (total_ns > top_ns) {
    const uint64_t other = total_ns - top_ns;
    std::fprintf(stderr,
                 "[sdpa_profile]   %-16s %9.3f ms %6.2f%%\n",
                 "top_other",
                 ns_to_ms(other),
                 100.0 * static_cast<double>(other) /
                     static_cast<double>(total_ns));
  }

  const uint64_t main_ns = state().ns[static_cast<int>(Slot::kMain)];
  const uint64_t inner_ns =
      state().ns[static_cast<int>(Slot::kInit)] +
      state().ns[static_cast<int>(Slot::kQPack)] +
      state().ns[static_cast<int>(Slot::kQkt)] +
      state().ns[static_cast<int>(Slot::kMask)] +
      state().ns[static_cast<int>(Slot::kSoftmax)] +
      state().ns[static_cast<int>(Slot::kPConvert)] +
      state().ns[static_cast<int>(Slot::kPv)] +
      state().ns[static_cast<int>(Slot::kFinalize)];
  if (main_ns > inner_ns) {
    const uint64_t other = main_ns - inner_ns;
    std::fprintf(stderr,
                 "[sdpa_profile]   %-16s %9.3f ms %6.2f%% of total\n",
                 "main_other",
                 ns_to_ms(other),
                 total_ns == 0 ? 0.0 :
                     100.0 * static_cast<double>(other) /
                         static_cast<double>(total_ns));
  }
}

}  // namespace fused_cpp::sdpa_profile

#define FUSED_CPP_SDPA_PROFILE_CONCAT_INNER(a, b) a##b
#define FUSED_CPP_SDPA_PROFILE_CONCAT(a, b) \
  FUSED_CPP_SDPA_PROFILE_CONCAT_INNER(a, b)
#define FUSED_CPP_SDPA_PROFILE_SCOPE(slot) \
  ::fused_cpp::sdpa_profile::ScopedTimer \
      FUSED_CPP_SDPA_PROFILE_CONCAT(_sdpa_profile_scope_, __LINE__)(slot)
#define FUSED_CPP_SDPA_PROFILE_DEEP_SCOPE(slot) \
  ::fused_cpp::sdpa_profile::ScopedTimer \
      FUSED_CPP_SDPA_PROFILE_CONCAT(_sdpa_profile_deep_scope_, __LINE__)( \
          slot, ::fused_cpp::sdpa_profile::deep_enabled())
