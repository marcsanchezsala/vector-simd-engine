// vector_ops.cpp
//
// Implementation notes:
//
// Rather than compiling separate translation units with different -march
// flags, each SIMD kernel is annotated with GCC/Clang's
// __attribute__((target("..."))) so a single TU can emit AVX2 and AVX-512
// code paths side by side while the rest of the program (and any caller)
// stays on the baseline ISA. This mirrors how real vector-DB engines
// (FAISS, hnswlib) do runtime dispatch: detect the CPU once, then always
// call through the fastest available kernel.
//
// All SIMD dot/L2 kernels reduce in the same tree-summation order regardless
// of dim, so results are deterministic per-kernel; small differences vs the
// scalar path (~1e-5 relative) are expected float-reassociation error, not
// bugs.

#include "vector_ops.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#define VECOPS_X86 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vecops {

// ---------------------------------------------------------------------------
// Aligned allocation
// ---------------------------------------------------------------------------
void AlignedDeleter::operator()(float* p) const noexcept {
#if defined(_WIN32)
    _aligned_free(p);   // MinGW and MSVC both route aligned allocs through
                         // the Windows CRT, which requires _aligned_free
                         // (plain free() on a _aligned_malloc pointer is UB).
#else
    std::free(p);
#endif
}

AlignedBuffer make_aligned(std::size_t count) {
    void* raw = nullptr;
    constexpr std::size_t kAlign = 64;
    std::size_t bytes = count * sizeof(float);
    if (bytes == 0) bytes = kAlign;
#if defined(_WIN32)
    raw = _aligned_malloc(bytes, kAlign);   // posix_memalign doesn't exist
                                             // on Windows (MinGW or MSVC)
    if (!raw) throw std::bad_alloc();
#else
    if (posix_memalign(&raw, kAlign, bytes) != 0) {
        throw std::bad_alloc();
    }
#endif
    std::memset(raw, 0, bytes);
    return AlignedBuffer(static_cast<float*>(raw));
}

// ---------------------------------------------------------------------------
// CPU feature detection
// ---------------------------------------------------------------------------
CpuFeatures detect_cpu_features() {
    CpuFeatures f;
#ifdef VECOPS_X86
    f.avx2    = __builtin_cpu_supports("avx2");
    f.fma     = __builtin_cpu_supports("fma");
    f.avx512f = __builtin_cpu_supports("avx512f");
#endif
    return f;
}

// ---------------------------------------------------------------------------
// Scalar baseline
// ---------------------------------------------------------------------------
// volatile-free, but compiled with -O2 -fno-tree-vectorize for this TU-wide
// setting is NOT what we want per-function, so instead we mark these with
// optimize("no-tree-vectorize") to guarantee a true scalar baseline even
// under -O3, giving an honest comparison against the hand-vectorized code.
__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize")))
float dot_scalar(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize")))
float l2sq_scalar(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize")))
float cosine_scalar(const float* a, const float* b, std::size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0f ? dot / denom : 0.0f;
}

#ifdef VECOPS_X86
// ---------------------------------------------------------------------------
// AVX2 + FMA kernels (8 x float32 per register)
// ---------------------------------------------------------------------------
__attribute__((target("avx2,fma")))
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

__attribute__((target("avx2,fma")))
float dot_avx2(const float* a, const float* b, std::size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum256(acc);
    for (; i < dim; ++i) sum += a[i] * b[i];  // tail (dim not a multiple of 8)
    return sum;
}

__attribute__((target("avx2,fma")))
float l2sq_avx2(const float* a, const float* b, std::size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    float sum = hsum256(acc);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

__attribute__((target("avx2,fma")))
float cosine_avx2(const float* a, const float* b, std::size_t dim) {
    __m256 acc_dot = _mm256_setzero_ps();
    __m256 acc_na  = _mm256_setzero_ps();
    __m256 acc_nb  = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
        acc_na  = _mm256_fmadd_ps(va, va, acc_na);
        acc_nb  = _mm256_fmadd_ps(vb, vb, acc_nb);
    }
    float dot = hsum256(acc_dot);
    float na  = hsum256(acc_na);
    float nb  = hsum256(acc_nb);
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0f ? dot / denom : 0.0f;
}

// ---------------------------------------------------------------------------
// AVX-512F kernels (16 x float32 per register)
// ---------------------------------------------------------------------------
__attribute__((target("avx512f")))
float dot_avx512(const float* a, const float* b, std::size_t dim) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx512f")))
float l2sq_avx512(const float* a, const float* b, std::size_t dim) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 d  = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

__attribute__((target("avx512f")))
float cosine_avx512(const float* a, const float* b, std::size_t dim) {
    __m512 acc_dot = _mm512_setzero_ps();
    __m512 acc_na  = _mm512_setzero_ps();
    __m512 acc_nb  = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc_dot = _mm512_fmadd_ps(va, vb, acc_dot);
        acc_na  = _mm512_fmadd_ps(va, va, acc_na);
        acc_nb  = _mm512_fmadd_ps(vb, vb, acc_nb);
    }
    float dot = _mm512_reduce_add_ps(acc_dot);
    float na  = _mm512_reduce_add_ps(acc_na);
    float nb  = _mm512_reduce_add_ps(acc_nb);
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0f ? dot / denom : 0.0f;
}

#else  // non-x86 fallback: alias SIMD entry points to scalar so the library
       // still links and behaves correctly (e.g. on ARM).
float dot_avx2(const float* a, const float* b, std::size_t dim)    { return dot_scalar(a, b, dim); }
float l2sq_avx2(const float* a, const float* b, std::size_t dim)   { return l2sq_scalar(a, b, dim); }
float cosine_avx2(const float* a, const float* b, std::size_t dim) { return cosine_scalar(a, b, dim); }
float dot_avx512(const float* a, const float* b, std::size_t dim)    { return dot_scalar(a, b, dim); }
float l2sq_avx512(const float* a, const float* b, std::size_t dim)   { return l2sq_scalar(a, b, dim); }
float cosine_avx512(const float* a, const float* b, std::size_t dim) { return cosine_scalar(a, b, dim); }
#endif

// ---------------------------------------------------------------------------
// Batch dispatch (OpenMP over rows, SIMD kernel per row)
// ---------------------------------------------------------------------------
namespace {

using Kernel = float (*)(const float*, const float*, std::size_t);

Kernel select_kernel(Metric metric, ImplKind impl) {
    switch (impl) {
        case ImplKind::Scalar:
            switch (metric) {
                case Metric::DotProduct: return dot_scalar;
                case Metric::L2Squared:  return l2sq_scalar;
                case Metric::Cosine:     return cosine_scalar;
            }
            break;
        case ImplKind::AVX2:
            switch (metric) {
                case Metric::DotProduct: return dot_avx2;
                case Metric::L2Squared:  return l2sq_avx2;
                case Metric::Cosine:     return cosine_avx2;
            }
            break;
        case ImplKind::AVX512:
            switch (metric) {
                case Metric::DotProduct: return dot_avx512;
                case Metric::L2Squared:  return l2sq_avx512;
                case Metric::Cosine:     return cosine_avx512;
            }
            break;
    }
    throw std::invalid_argument("unknown metric/impl combination");
}

}  // namespace

void batch_compute(Metric metric, ImplKind impl,
                    const float* query,
                    const float* database, std::size_t n_vectors, std::size_t dim,
                    float* results,
                    int num_threads) {
    Kernel kernel = select_kernel(metric, impl);

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t row = 0; row < n_vectors; ++row) {
        const float* vec = database + row * dim;
        results[row] = kernel(query, vec, dim);
    }
}

}  // namespace vecops
