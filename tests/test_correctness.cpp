// test_correctness.cpp
// Verifies AVX2 / AVX-512 kernels agree with the scalar baseline, including
// dims that are NOT exact multiples of 8 or 16 (exercises the tail loop).

#include "vector_ops.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::printf("  FAIL: %s\n", msg);
        ++g_failures;
    } else {
        std::printf("  ok:   %s\n", msg);
    }
}

void test_dim(std::size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(dim), b(dim);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    float d_s = vecops::dot_scalar(a.data(), b.data(), dim);
    float d_2 = vecops::dot_avx2(a.data(), b.data(), dim);
    float l_s = vecops::l2sq_scalar(a.data(), b.data(), dim);
    float l_2 = vecops::l2sq_avx2(a.data(), b.data(), dim);
    float c_s = vecops::cosine_scalar(a.data(), b.data(), dim);
    float c_2 = vecops::cosine_avx2(a.data(), b.data(), dim);

    char msg[128];
    std::snprintf(msg, sizeof(msg), "dim=%zu dot scalar~avx2 (%.6f vs %.6f)", dim, d_s, d_2);
    check(std::fabs(d_s - d_2) < 1e-3f, msg);

    std::snprintf(msg, sizeof(msg), "dim=%zu l2sq scalar~avx2 (%.6f vs %.6f)", dim, l_s, l_2);
    check(std::fabs(l_s - l_2) < 1e-3f, msg);

    std::snprintf(msg, sizeof(msg), "dim=%zu cosine scalar~avx2 (%.6f vs %.6f)", dim, c_s, c_2);
    check(std::fabs(c_s - c_2) < 1e-4f, msg);

    vecops::CpuFeatures cpu = vecops::detect_cpu_features();
    if (cpu.avx512f) {
        float d_5 = vecops::dot_avx512(a.data(), b.data(), dim);
        float l_5 = vecops::l2sq_avx512(a.data(), b.data(), dim);
        float c_5 = vecops::cosine_avx512(a.data(), b.data(), dim);

        std::snprintf(msg, sizeof(msg), "dim=%zu dot scalar~avx512 (%.6f vs %.6f)", dim, d_s, d_5);
        check(std::fabs(d_s - d_5) < 1e-3f, msg);

        std::snprintf(msg, sizeof(msg), "dim=%zu l2sq scalar~avx512 (%.6f vs %.6f)", dim, l_s, l_5);
        check(std::fabs(l_s - l_5) < 1e-3f, msg);

        std::snprintf(msg, sizeof(msg), "dim=%zu cosine scalar~avx512 (%.6f vs %.6f)", dim, c_s, c_5);
        check(std::fabs(c_s - c_5) < 1e-4f, msg);
    }
}

void test_batch() {
    const std::size_t dim = 768, n = 2000;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto query = vecops::make_aligned(dim);
    auto db = vecops::make_aligned(n * dim);
    for (std::size_t i = 0; i < dim; ++i) query[i] = dist(rng);
    for (std::size_t i = 0; i < n * dim; ++i) db[i] = dist(rng);

    std::vector<float> r_scalar(n), r_avx2(n);
    vecops::batch_compute(vecops::Metric::Cosine, vecops::ImplKind::Scalar,
                           query.get(), db.get(), n, dim, r_scalar.data(), 1);
    vecops::batch_compute(vecops::Metric::Cosine, vecops::ImplKind::AVX2,
                           query.get(), db.get(), n, dim, r_avx2.data(), 0);

    double max_diff = 0.0;
    for (std::size_t i = 0; i < n; ++i) max_diff = std::max(max_diff, (double)std::fabs(r_scalar[i] - r_avx2[i]));
    char msg[128];
    std::snprintf(msg, sizeof(msg), "batch cosine scalar~avx2 max_diff=%.3e", max_diff);
    check(max_diff < 1e-4, msg);
}

}  // namespace

int main() {
    std::mt19937 rng(123);
    // exact multiples of 8/16, and non-multiples to exercise tail loops
    for (std::size_t dim : {8, 16, 768, 770, 763, 1, 7, 17}) {
        std::printf("Testing dim=%zu\n", dim);
        test_dim(dim, rng);
    }
    std::printf("Testing batch_compute\n");
    test_batch();

    if (g_failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) FAILED.\n", g_failures);
    return 1;
}
