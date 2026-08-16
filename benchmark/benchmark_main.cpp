// benchmark_main.cpp
//
// Benchmarks scalar vs AVX2 vs AVX-512 kernels for Dot Product, L2 (squared)
// distance, and Cosine Similarity, on N 768-dim float32 vectors, both
// single-threaded and OpenMP-parallel.

#include "vector_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;

namespace {

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void fill_random(float* buf, std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

struct Result {
    std::string label;
    double ms;
    double vectors_per_sec;
};

Result run_batch(const char* label, vecops::Metric metric, vecops::ImplKind impl,
                  const float* query, const float* db, std::size_t n, std::size_t dim,
                  float* results, int threads) {
    auto t0 = Clock::now();
    vecops::batch_compute(metric, impl, query, db, n, dim, results, threads);
    auto t1 = Clock::now();
    double ms = elapsed_ms(t0, t1);
    return Result{label, ms, n / (ms / 1000.0)};
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

void print_row(const Result& r, double speedup) {
    std::printf("  %-28s %10.2f ms   %14.0f vec/s   %6.2fx\n",
                r.label.c_str(), r.ms, r.vectors_per_sec, speedup);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t n_vectors = 1'000'000;
    std::size_t dim = 768;
    if (argc > 1) n_vectors = std::stoul(argv[1]);
    if (argc > 2) dim = std::stoul(argv[2]);

    vecops::CpuFeatures cpu = vecops::detect_cpu_features();
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif

    std::printf("=== SIMD-Accelerated Vector Similarity Engine ===\n");
    std::printf("Vectors: %zu   Dim: %zu   (%.1f MB dataset)\n",
                n_vectors, dim, (double)(n_vectors * dim * sizeof(float)) / (1024 * 1024));
    std::printf("CPU features: AVX2=%s FMA=%s AVX-512F=%s\n",
                cpu.avx2 ? "yes" : "no", cpu.fma ? "yes" : "no", cpu.avx512f ? "yes" : "no");
    std::printf("OpenMP max threads: %d\n\n", max_threads);

    std::mt19937 rng(42);
    auto query = vecops::make_aligned(dim);
    auto db    = vecops::make_aligned(n_vectors * dim);
    fill_random(query.get(), dim, rng);
    fill_random(db.get(), n_vectors * dim, rng);

    std::vector<float> res_scalar(n_vectors), res_avx2(n_vectors), res_avx512(n_vectors);

    struct MetricSpec { const char* name; vecops::Metric metric; };
    std::vector<MetricSpec> metrics = {
        {"Dot Product", vecops::Metric::DotProduct},
        {"L2 Squared Distance", vecops::Metric::L2Squared},
        {"Cosine Similarity", vecops::Metric::Cosine},
    };

    for (auto& spec : metrics) {
        std::printf("--- %s ---\n", spec.name);

        auto r_scalar = run_batch("scalar (1 thread)", spec.metric, vecops::ImplKind::Scalar,
                                   query.get(), db.get(), n_vectors, dim, res_scalar.data(), 1);
        print_row(r_scalar, 1.0);

        auto r_avx2_1 = run_batch("AVX2+FMA (1 thread)", spec.metric, vecops::ImplKind::AVX2,
                                   query.get(), db.get(), n_vectors, dim, res_avx2.data(), 1);
        print_row(r_avx2_1, r_scalar.ms / r_avx2_1.ms);

        if (cpu.avx512f) {
            auto r_avx512_1 = run_batch("AVX-512 (1 thread)", spec.metric, vecops::ImplKind::AVX512,
                                         query.get(), db.get(), n_vectors, dim, res_avx512.data(), 1);
            print_row(r_avx512_1, r_scalar.ms / r_avx512_1.ms);
        }

        if (max_threads > 1) {
            auto r_avx2_n = run_batch(("AVX2+FMA (" + std::to_string(max_threads) + " threads)").c_str(),
                                       spec.metric, vecops::ImplKind::AVX2,
                                       query.get(), db.get(), n_vectors, dim, res_avx2.data(), max_threads);
            print_row(r_avx2_n, r_scalar.ms / r_avx2_n.ms);

            if (cpu.avx512f) {
                auto r_avx512_n = run_batch(("AVX-512 (" + std::to_string(max_threads) + " threads)").c_str(),
                                             spec.metric, vecops::ImplKind::AVX512,
                                             query.get(), db.get(), n_vectors, dim, res_avx512.data(), max_threads);
                print_row(r_avx512_n, r_scalar.ms / r_avx512_n.ms);
            }
        } else {
            std::printf("  (only 1 CPU core available in this environment — multi-thread\n"
                        "   speedup will be minimal/negative here; re-run on a multi-core\n"
                        "   machine to see OpenMP scaling.)\n");
            // still exercise the multithreaded code path for correctness
            run_batch("AVX2+FMA (omp, 1 core)", spec.metric, vecops::ImplKind::AVX2,
                      query.get(), db.get(), n_vectors, dim, res_avx2.data(), 0);
        }

        double diff_avx2 = max_abs_diff(res_scalar, res_avx2);
        std::printf("  max |scalar - AVX2|   = %.3e\n", diff_avx2);
        if (cpu.avx512f) {
            double diff_avx512 = max_abs_diff(res_scalar, res_avx512);
            std::printf("  max |scalar - AVX512| = %.3e\n", diff_avx512);
        }
        std::printf("\n");
    }

    std::printf("Done.\n");
    return 0;
}
