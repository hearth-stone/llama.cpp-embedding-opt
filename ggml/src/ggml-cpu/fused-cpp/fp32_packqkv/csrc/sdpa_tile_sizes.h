#pragma once
// ── 三级 cache 常量 / tile 尺寸推导（共享头）─────────────────────────────
//
// 本头文件由
//   * csrc/sdpa_flash2_neon_cache.cpp
//   * csrc/sdpa_flash2_neon_l3kv.cpp
// 共同包含，提供：
//   1. 编译期 L1/L2/L3 字节数与使用比例宏（允许 `-D` 覆写）；
//   2. 运行时 cache 大小探测（macOS sysctl / Linux sysconf+/sys 扫描），
//      首次调用一次，结果缓存在 inline 函数局部 static 变量中；
//   3. `TileSizes` 结构体 + `compute_tile_sizes()`：原 flash2_neon_cache
//      的纯函数推导；
//   4. `compute_tile_sizes_l3kv()`：在 (3) 的基础上按 Ev 自适应收紧
//      `Lc_l2`，避免 huge head_dim 场景下 O 累加器爆 L2。
//
// 设计要点：
//   * 所有函数都用 `inline`，便于多个 TU 包含；局部 static 变量在 inline
//     函数中跨 TU 共享存储（C++17 magic statics 保证一次性初始化）。
//   * 不依赖任何外部 tensor framework 头，仅 <cstdint> / <cstdio> /
//     <array> 等标准头 + 平台 sysctl/sysconf。
//   * 修改本文件需同时考虑 cache 与 l3kv 两个内核行为是否仍一致；新增
//     字段 SHALL 默认 0 / SHALL 不破坏 `compute_tile_sizes` 的契约。

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

// ──────────────────────────────────────────────────────────────────────
// L1 / L2 / L3 字节数与使用比例（编译期宏，允许 `-D` 覆写）
// ──────────────────────────────────────────────────────────────────────

#ifndef FUSED_CPP_SDPA_L1_BYTES
#define FUSED_CPP_SDPA_L1_BYTES 65536          // 64 KB
#define FUSED_CPP_SDPA_L1_BYTES_USER_OVERRIDE 0
#else
#define FUSED_CPP_SDPA_L1_BYTES_USER_OVERRIDE 1
#endif
#ifndef FUSED_CPP_SDPA_L2_BYTES
#define FUSED_CPP_SDPA_L2_BYTES 1310720        // 1280 KB
#define FUSED_CPP_SDPA_L2_BYTES_USER_OVERRIDE 0
#else
#define FUSED_CPP_SDPA_L2_BYTES_USER_OVERRIDE 1
#endif
#ifndef FUSED_CPP_SDPA_L3_BYTES
#define FUSED_CPP_SDPA_L3_BYTES 73400320       // 70 MB
#define FUSED_CPP_SDPA_L3_BYTES_USER_OVERRIDE 0
#else
#define FUSED_CPP_SDPA_L3_BYTES_USER_OVERRIDE 1
#endif

#ifndef FUSED_CPP_SDPA_DISABLE_CACHE_PROBE
#define FUSED_CPP_SDPA_DISABLE_CACHE_PROBE 0
#endif

#ifndef FUSED_CPP_SDPA_L1_RATIO
#define FUSED_CPP_SDPA_L1_RATIO 0.50
#endif
#ifndef FUSED_CPP_SDPA_L2_RATIO
#define FUSED_CPP_SDPA_L2_RATIO 0.75
#endif
#ifndef FUSED_CPP_SDPA_L3_RATIO
#define FUSED_CPP_SDPA_L3_RATIO 0.70
#endif

static_assert(FUSED_CPP_SDPA_L1_RATIO > 0.0 && FUSED_CPP_SDPA_L1_RATIO <= 1.0,
              "FUSED_CPP_SDPA_L1_RATIO must be in (0.0, 1.0]");
static_assert(FUSED_CPP_SDPA_L2_RATIO > 0.0 && FUSED_CPP_SDPA_L2_RATIO <= 1.0,
              "FUSED_CPP_SDPA_L2_RATIO must be in (0.0, 1.0]");
static_assert(FUSED_CPP_SDPA_L3_RATIO > 0.0 && FUSED_CPP_SDPA_L3_RATIO <= 1.0,
              "FUSED_CPP_SDPA_L3_RATIO must be in (0.0, 1.0]");
static_assert(FUSED_CPP_SDPA_L1_BYTES > 0,
              "FUSED_CPP_SDPA_L1_BYTES must be positive");
static_assert(FUSED_CPP_SDPA_L2_BYTES > 0,
              "FUSED_CPP_SDPA_L2_BYTES must be positive");
static_assert(FUSED_CPP_SDPA_L3_BYTES > 0,
              "FUSED_CPP_SDPA_L3_BYTES must be positive");

namespace fused_cpp::sdpa_tile_sizes {

// ──────────────────────────────────────────────────────────────────────
// 单级 cache 字节数探测；探测失败返回 -1。
// ──────────────────────────────────────────────────────────────────────

inline int64_t probe_cache_bytes_one_level(int level) {
#if FUSED_CPP_SDPA_DISABLE_CACHE_PROBE
  (void)level;
  return -1;
#else

#if defined(__APPLE__)
  // Apple Silicon 是 big.LITTLE：P-core (perflevel0) 与 E-core (perflevel1)
  // 各有独立的 L1d/L2 容量。`hw.l1dcachesize` / `hw.l2cachesize` 这种「无后缀」
  // 的 key 在 macOS 上**默认返回 E-core 的容量**（= perflevel1），但单线程
  // 高性能 SDPA 实际跑在 P-core 上，要用 perflevel0 的容量来算 tile size，
  // 否则会按 E-core 64 KB / 4 MB 来定 working set，浪费 P-core 的 128 KB
  // / 16 MB cache 一半容量。
  //
  // 修复：探测时把 `hw.perflevel0.<level>cachesize` 放在**第一优先级**，
  // 拿不到（旧 macOS 或 Intel mac）才回退到无后缀 key。
  const char* keys_apple[3][2] = {
      {"hw.perflevel0.l1dcachesize", "hw.l1dcachesize"},
      {"hw.perflevel0.l2cachesize",  "hw.l2cachesize"},
      {"hw.perflevel0.l3cachesize",  "hw.l3cachesize"},
  };
  if (level < 1 || level > 3) return -1;
  for (int i = 0; i < 2; ++i) {
    const char* key = keys_apple[level - 1][i];
    if (key == nullptr) continue;
    uint64_t value = 0;
    size_t len = sizeof(value);
    if (sysctlbyname(key, &value, &len, nullptr, 0) == 0 && value > 0) {
      return static_cast<int64_t>(value);
    }
  }
  return -1;

#elif defined(__linux__)
  long sc_value = -1;
  switch (level) {
    case 1:
#if defined(_SC_LEVEL1_DCACHE_SIZE)
      sc_value = sysconf(_SC_LEVEL1_DCACHE_SIZE);
#endif
      break;
    case 2:
#if defined(_SC_LEVEL2_CACHE_SIZE)
      sc_value = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
      break;
    case 3:
#if defined(_SC_LEVEL3_CACHE_SIZE)
      sc_value = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
      break;
    default:
      return -1;
  }
  if (sc_value > 0) {
    return static_cast<int64_t>(sc_value);
  }

  for (int idx = 0; idx < 16; ++idx) {
    char path_level[256];
    char path_type[256];
    char path_size[256];
    std::snprintf(path_level, sizeof(path_level),
                  "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
    std::snprintf(path_type, sizeof(path_type),
                  "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
    std::snprintf(path_size, sizeof(path_size),
                  "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);

    FILE* fp_level = std::fopen(path_level, "r");
    if (fp_level == nullptr) {
      break;
    }
    int parsed_level = -1;
    if (std::fscanf(fp_level, "%d", &parsed_level) != 1) parsed_level = -1;
    std::fclose(fp_level);
    if (parsed_level != level) continue;

    FILE* fp_type = std::fopen(path_type, "r");
    if (fp_type == nullptr) continue;
    char type_buf[32] = {0};
    size_t nread = std::fread(type_buf, 1, sizeof(type_buf) - 1, fp_type);
    std::fclose(fp_type);
    (void)nread;
    for (size_t i = 0; i < sizeof(type_buf); ++i) {
      if (type_buf[i] == '\n' || type_buf[i] == '\r') { type_buf[i] = '\0'; break; }
    }
    if (std::strcmp(type_buf, "Data") != 0 &&
        std::strcmp(type_buf, "Unified") != 0) {
      continue;
    }

    FILE* fp_size = std::fopen(path_size, "r");
    if (fp_size == nullptr) continue;
    char size_buf[64] = {0};
    if (std::fgets(size_buf, sizeof(size_buf), fp_size) == nullptr) {
      std::fclose(fp_size);
      continue;
    }
    std::fclose(fp_size);

    long long num = 0;
    char suffix = '\0';
    int matched = std::sscanf(size_buf, "%lld%c", &num, &suffix);
    if (matched < 1 || num <= 0) continue;
    int64_t bytes = static_cast<int64_t>(num);
    switch (suffix) {
      case 'K': case 'k': bytes *= 1024;             break;
      case 'M': case 'm': bytes *= 1024 * 1024;      break;
      case 'G': case 'g': bytes *= 1024 * 1024 * 1024LL; break;
      default: /* 纯字节 */ break;
    }
    if (bytes > 0) {
      return bytes;
    }
  }
  return -1;

#else
  (void)level;
  return -1;
#endif

#endif  // FUSED_CPP_SDPA_DISABLE_CACHE_PROBE
}

// 三级 cache 字节数（首次调用时探测，之后直接返回缓存值）。
// 调用方按 level=1/2/3 索引（level-1）。
inline const std::array<int64_t, 3>& effective_cache_bytes() {
  static const std::array<int64_t, 3> kBytes = []() {
    std::array<int64_t, 3> result = {
        static_cast<int64_t>(FUSED_CPP_SDPA_L1_BYTES),
        static_cast<int64_t>(FUSED_CPP_SDPA_L2_BYTES),
        static_cast<int64_t>(FUSED_CPP_SDPA_L3_BYTES),
    };
    const int kUserOverride[3] = {
        FUSED_CPP_SDPA_L1_BYTES_USER_OVERRIDE,
        FUSED_CPP_SDPA_L2_BYTES_USER_OVERRIDE,
        FUSED_CPP_SDPA_L3_BYTES_USER_OVERRIDE,
    };
    int64_t probed[3] = {-1, -1, -1};
    for (int level = 1; level <= 3; ++level) {
      if (kUserOverride[level - 1]) continue;
      probed[level - 1] = probe_cache_bytes_one_level(level);
      if (probed[level - 1] > 0) {
        result[level - 1] = probed[level - 1];
      }
    }

    const char* dbg = std::getenv("FUSED_CPP_SDPA_DEBUG_TILE");
    if (dbg != nullptr && std::strcmp(dbg, "1") == 0) {
      std::fprintf(stderr,
          "[sdpa_tile_sizes] effective L1/L2/L3 bytes = "
          "%lld / %lld / %lld  (override=%d/%d/%d, probed=%lld/%lld/%lld)\n",
          static_cast<long long>(result[0]),
          static_cast<long long>(result[1]),
          static_cast<long long>(result[2]),
          kUserOverride[0], kUserOverride[1], kUserOverride[2],
          static_cast<long long>(probed[0]),
          static_cast<long long>(probed[1]),
          static_cast<long long>(probed[2]));
    }
    return result;
  }();
  return kBytes;
}

// ──────────────────────────────────────────────────────────────────────
// 整数算术 helper
// ──────────────────────────────────────────────────────────────────────

inline int64_t ceil_div_pos(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int64_t floor_to_mult_min(int64_t x, int64_t m) {
  if (x < m) return m;
  return (x / m) * m;
}

// ──────────────────────────────────────────────────────────────────────
// TileSizes & 推导函数
// ──────────────────────────────────────────────────────────────────────

struct TileSizes {
  int64_t Sc_l3;
  int64_t Sc_l2;
  int64_t Lc_l2;
  int64_t Sk_micro;
  int64_t Lq_micro;
  int64_t Ev_micro;
};

// compute_tile_sizes：与 sdpa_flash2_neon_cache.cpp 历史版本完全一致的
// 推导。Lc_l2 固定 64（或 floor_to_mult_min(L, 8)）。
inline TileSizes compute_tile_sizes(
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t L,
    int64_t E,
    int64_t Ev,
    int64_t sizeof_elt) {
  (void)B;
  (void)N;

  constexpr int64_t Sk_micro = 8;
  constexpr int64_t Lq_micro = 8;
  constexpr int64_t Ev_micro = 8;

  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l3_budget =
      static_cast<int64_t>(cache_bytes[2] * FUSED_CPP_SDPA_L3_RATIO);
  const int64_t l2_budget =
      static_cast<int64_t>(cache_bytes[1] * FUSED_CPP_SDPA_L2_RATIO);
  const int64_t l1_budget =
      static_cast<int64_t>(cache_bytes[0] * FUSED_CPP_SDPA_L1_RATIO);
  (void)l1_budget;

  int64_t Lc_l2_cand = 64;
  if (L < Lc_l2_cand) {
    Lc_l2_cand = floor_to_mult_min(L, Lq_micro);
  }
  int64_t Lc_l2 = std::max<int64_t>(Lc_l2_cand, Lq_micro);

  const int64_t l2_const_bytes =
      Lc_l2 * E * sizeof_elt
      + Lc_l2 * Ev * 4
      + Lc_l2 * 2 * 4;

  const int64_t per_sc =
      2 * (E + Ev) * sizeof_elt + Lc_l2 * 4;
  const int64_t l2_remain = std::max<int64_t>(
      0, l2_budget - l2_const_bytes);

  int64_t Sc_l2_cap;
  if (per_sc <= 0) {
    Sc_l2_cap = std::max<int64_t>(Sk_micro, S);
  } else {
    Sc_l2_cap = l2_remain / per_sc;
  }
  if (Sc_l2_cap < Sk_micro) {
    Sc_l2_cap = Sk_micro;
  }
  int64_t Sc_l2 = floor_to_mult_min(std::min(Sc_l2_cap, S), Sk_micro);
  if (Sc_l2 < Sk_micro) {
    Sc_l2 = Sk_micro;
  }

  const int64_t kv_per_head_bytes = S * (E + Ev) * sizeof_elt;
  int64_t Sc_l3;
  if (kv_per_head_bytes <= l3_budget) {
    Sc_l3 = S;
  } else {
    int64_t Sc_l3_cap = l3_budget / std::max<int64_t>(1, (E + Ev) * sizeof_elt);
    if (Sc_l3_cap < Sc_l2) {
      Sc_l3_cap = Sc_l2;
    }
    Sc_l3 = (Sc_l3_cap / Sc_l2) * Sc_l2;
    if (Sc_l3 < Sc_l2) Sc_l3 = Sc_l2;
    if (Sc_l3 > S) Sc_l3 = S;
  }
  if (Sc_l3 < Sc_l2) {
    Sc_l3 = Sc_l2;
  } else {
    int64_t aligned = (Sc_l3 / Sc_l2) * Sc_l2;
    if (aligned < Sc_l2) aligned = Sc_l2;
    Sc_l3 = aligned;
  }
  if (Sc_l3 > S) Sc_l3 = S;

  if (S > 0 && Sc_l3 < Sc_l2) {
    Sc_l2 = Sc_l3;
  }

  TileSizes ts{};
  ts.Sc_l3 = Sc_l3;
  ts.Sc_l2 = Sc_l2;
  ts.Lc_l2 = Lc_l2;
  ts.Sk_micro = Sk_micro;
  ts.Lq_micro = Lq_micro;
  ts.Ev_micro = Ev_micro;

#ifdef FUSED_CPP_SDPA_DEBUG_TILES
  std::fprintf(stderr,
      "[sdpa_tile_sizes] tiles: B=%lld N=%lld S=%lld L=%lld E=%lld "
      "Ev=%lld sizeof_elt=%lld -> "
      "Sc_l3=%lld Sc_l2=%lld Lc_l2=%lld Sk_micro=%lld Lq_micro=%lld "
      "Ev_micro=%lld\n",
      (long long)B, (long long)N, (long long)S, (long long)L,
      (long long)E, (long long)Ev, (long long)sizeof_elt,
      (long long)ts.Sc_l3, (long long)ts.Sc_l2, (long long)ts.Lc_l2,
      (long long)ts.Sk_micro, (long long)ts.Lq_micro, (long long)ts.Ev_micro);
#endif

  return ts;
}

// compute_tile_sizes_l3kv：l3kv 内核专用。在原 compute_tile_sizes 基础上
// **按 Ev 自适应收紧 Lc_l2**，避免 huge head_dim 场景下 O 累加器爆 L2：
//
//   bytes_per_row(Lc) = Q tile + O acc(fp32) + running m/l + scores+P_hat(fp32)
//                     ≈ E * sizeof_elt + 4 * Ev + 2 * 4 + 2 * Sc_l2 * 4
//
// 但 Sc_l2 在 compute_tile_sizes 里已经按 Lc_l2=64 估算过了。l3kv 把 K/V
// tile 在所有 Lc_l2/8 个 inner 8 行组之间复用，因此 scores / P_hat 缓冲
// 需要 Lc_l2 * Sc_l2 而不是 8 * Sc_l2。所以这里按更严格的口径重算 Lc：
//
//   per_row_bytes ≈ sizeof_elt * E         // Q tile
//                 + 4 * Ev                  // O acc fp32
//                 + 2 * 4                   // running max/sum
//                 + 2 * Sc_l2 * 4           // scores + P_hat fp32（per row）
//
// Lc_cap = (l2_budget * 0.5) / per_row_bytes，clamp 到 [8, 64] 且 8 倍数。
inline TileSizes compute_tile_sizes_l3kv(
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t L,
    int64_t E,
    int64_t Ev,
    int64_t sizeof_elt) {
  TileSizes ts = compute_tile_sizes(B, N, S, L, E, Ev, sizeof_elt);

  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l2_budget =
      static_cast<int64_t>(cache_bytes[1] * FUSED_CPP_SDPA_L2_RATIO);

  // 给 l3kv 内核的 accumulator 用一半 L2 预算（剩下一半给双缓冲 K/V tile
  // + L2 const overhead）。
  const int64_t f_l2 = l2_budget / 2;

  // 注意：scores + P_hat = 2 * Sc_l2 * 4 / row。
  const int64_t per_row_bytes =
      sizeof_elt * E
      + 4 * Ev
      + 2 * 4
      + 2 * ts.Sc_l2 * 4;
  if (per_row_bytes <= 0) {
    return ts;
  }

  int64_t Lc_cap = f_l2 / per_row_bytes;
  // 8 倍数且夹紧到 [8, 64]。
  Lc_cap = (Lc_cap / 8) * 8;
  if (Lc_cap < 8) Lc_cap = 8;
  if (Lc_cap > 64) Lc_cap = 64;

  // 不超过 L 自身的 8 倍数对齐。
  int64_t Lc_by_L = floor_to_mult_min(L, 8);

  int64_t Lc_new = std::min<int64_t>(Lc_cap, Lc_by_L);
  // 至少 8（micro-kernel 步长）。
  if (Lc_new < 8) Lc_new = 8;

  ts.Lc_l2 = Lc_new;

#ifdef FUSED_CPP_SDPA_DEBUG_TILES
  std::fprintf(stderr,
      "[sdpa_tile_sizes] l3kv refined: per_row_bytes=%lld Lc_cap=%lld -> "
      "Lc_l2=%lld (Sc_l2=%lld Sc_l3=%lld)\n",
      (long long)per_row_bytes, (long long)Lc_cap, (long long)ts.Lc_l2,
      (long long)ts.Sc_l2, (long long)ts.Sc_l3);
#endif

  return ts;
}

}  // namespace fused_cpp::sdpa_tile_sizes
