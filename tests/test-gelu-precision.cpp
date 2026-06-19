#include "ggml-cpu.h"
#include "ggml-cpu/vec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

struct diff_stats {
    double mean_abs = 0.0;
    double rmse = 0.0;
    double p50_abs = 0.0;
    double p95_abs = 0.0;
    double p99_abs = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
};

static diff_stats calc_diff_stats(const std::vector<float> & ref, const std::vector<float> & val) {
    diff_stats st;
    std::vector<double> abs_diff;
    abs_diff.reserve(ref.size());

    double sum_abs = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = std::abs((double) val[i] - (double) ref[i]);
        const double denom = std::max(std::abs((double) ref[i]), 1e-6);
        abs_diff.push_back(diff);
        sum_abs += diff;
        sum_sq += diff*diff;
        st.max_abs = std::max(st.max_abs, diff);
        st.max_rel = std::max(st.max_rel, diff/denom);
    }

    std::sort(abs_diff.begin(), abs_diff.end());

    const size_t n = abs_diff.size();
    st.mean_abs = sum_abs/(double) n;
    st.rmse = std::sqrt(sum_sq/(double) n);
    st.p50_abs = abs_diff[(size_t) (0.50*(double) (n - 1))];
    st.p95_abs = abs_diff[(size_t) (0.95*(double) (n - 1))];
    st.p99_abs = abs_diff[(size_t) (0.99*(double) (n - 1))];

    return st;
}

static void print_stats(const char * name, const char * compare, const diff_stats & st, size_t n) {
    std::printf(
        "case=%s compare=%s n=%zu mean_abs=%.9g rmse=%.9g p50_abs=%.9g p95_abs=%.9g p99_abs=%.9g max_abs=%.9g max_rel=%.9g\n",
        name, compare, n, st.mean_abs, st.rmse, st.p50_abs, st.p95_abs, st.p99_abs, st.max_abs, st.max_rel);
}

static void run_case(const char * name, const std::vector<float> & x) {
    std::vector<float> fp32(x.size());
    std::vector<float> table(x.size());
    std::vector<float> table_f32(x.size());
    std::vector<float> fp16(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        fp32[i] = ggml_gelu_f32(x[i]);
        table[i] = ggml_gelu_f32_table(x[i]);
        table_f32[i] = ggml_gelu_f32_table_f32(x[i]);
        fp16[i] = ggml_gelu_f32_fp16(x[i]);
    }

    print_stats(name, "fp16_direct_vs_table", calc_diff_stats(table, fp16), x.size());
    print_stats(name, "fp16_direct_vs_fp32",  calc_diff_stats(fp32,  fp16), x.size());
    print_stats(name, "fp16_table_vs_fp32",   calc_diff_stats(fp32,  table), x.size());
    print_stats(name, "f32_table_vs_table",   calc_diff_stats(table, table_f32), x.size());
    print_stats(name, "f32_table_vs_fp32",    calc_diff_stats(fp32,  table_f32), x.size());
}

static double checksum(const std::vector<float> & x) {
    double sum = 0.0;
    for (float v : x) {
        sum += v;
    }
    return sum;
}

static std::vector<float> make_uniform(size_t n, float lo, float hi, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> x(n);
    for (float & v : x) {
        v = dist(rng);
    }
    return x;
}

static std::vector<float> make_normal(size_t n, float mean, float stddev, std::mt19937 & rng) {
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> x(n);
    for (float & v : x) {
        v = dist(rng);
    }
    return x;
}

static void run_bench(size_t n, int iters, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> x = make_normal(n, 0.0f, 1.0f, rng);
    std::vector<float> y(n);

    ggml_vec_gelu_f32((int) n, y.data(), x.data());

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_vec_gelu_f32((int) n, y.data(), x.data());
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double elems = (double) n*(double) iters;
    const char * mode = getenv("GGML_GELU_MODE");
    const char * table = getenv("GGML_GELU_TABLE");
    const char * unroll = getenv("GGML_GELU_UNROLL");
    const char * sve_unroll = getenv("GGML_GELU_SVE_UNROLL");

    std::printf(
        "bench mode=%s table=%s unroll=%s sve_unroll=%s n=%zu iters=%d total_ms=%.6f ns_per_elem=%.6f elems_per_s=%.6f checksum=%.9g\n",
        mode ? mode : "", table ? table : "", unroll ? unroll : "", sve_unroll ? sve_unroll : "", n, iters, total_ms,
        total_ms*1.0e6/elems, elems/(total_ms*1.0e-3), checksum(y));
}

int main(int argc, char ** argv) {
    size_t n = 1 << 20;
    int iters = 100;
    uint32_t seed = 1234;
    bool bench = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) {
            n = (size_t) std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
        } else if (arg == "--bench") {
            bench = true;
        } else {
            std::fprintf(stderr, "usage: %s [--bench] [--n N] [--iters ITERS] [--seed SEED]\n", argv[0]);
            return 1;
        }
    }

    ggml_cpu_init();

    if (bench) {
        run_bench(n, iters, seed);
        return 0;
    }

    std::mt19937 rng(seed);
    run_case("uniform_-10_10", make_uniform(n, -10.0f, 10.0f, rng));
    run_case("normal_0_1",     make_normal(n, 0.0f, 1.0f, rng));
    run_case("normal_0_3",     make_normal(n, 0.0f, 3.0f, rng));

    return 0;
}
