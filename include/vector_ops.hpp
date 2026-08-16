// vector_ops.hpp
// SIMD-accelerated vector similarity engine for high-dimensional AI embeddings.
//
// Provides three metric families (dot product, L2 distance, cosine similarity)
// each with three implementations:
//   - scalar     : plain C++ loop (auto-vectorization disabled, baseline)
//   - avx2       : hand-written AVX2 + FMA intrinsics (8 floats/register)
//   - avx512     : hand-written AVX-512F intrinsics (16 floats/register)
//
// Batch variants compute one query against N database vectors and are
// parallelized across rows with OpenMP on top of the SIMD kernels.
//
// All vectors are assumed to be contiguous float32 arrays of length `dim`.
// For best SIMD throughput, allocate with 64-byte alignment (see
// AlignedBuffer below) so loads can use aligned instructions where used.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace vecops {

// ---------------------------------------------------------------------------
// Aligned allocation helper (64-byte alignment, suitable for AVX-512 as well
// as AVX2 / cache-line alignment).
// ---------------------------------------------------------------------------
struct AlignedDeleter {
    void operator()(float* p) const noexcept;
};
using AlignedBuffer = std::unique_ptr<float[], AlignedDeleter>;

// Allocates `count` floats aligned to 64 bytes, zero-initialized.
AlignedBuffer make_aligned(std::size_t count);

// ---------------------------------------------------------------------------
// Feature detection (runtime CPUID check, not just compile-time flags)
// ---------------------------------------------------------------------------
struct CpuFeatures {
    bool avx2   = false;
    bool fma    = false;
    bool avx512f = false;
};
CpuFeatures detect_cpu_features();

// ---------------------------------------------------------------------------
// Single-pair kernels
// ---------------------------------------------------------------------------

// Dot product: sum(a[i] * b[i])
float dot_scalar(const float* a, const float* b, std::size_t dim);
float dot_avx2(const float* a, const float* b, std::size_t dim);
float dot_avx512(const float* a, const float* b, std::size_t dim);

// Squared L2 distance: sum((a[i]-b[i])^2)   [no sqrt, cheaper & monotonic]
float l2sq_scalar(const float* a, const float* b, std::size_t dim);
float l2sq_avx2(const float* a, const float* b, std::size_t dim);
float l2sq_avx512(const float* a, const float* b, std::size_t dim);

// L2 (Euclidean) distance: sqrt(l2sq)
inline float l2_scalar(const float* a, const float* b, std::size_t dim);
inline float l2_avx2(const float* a, const float* b, std::size_t dim);
inline float l2_avx512(const float* a, const float* b, std::size_t dim);

// Cosine similarity: dot(a,b) / (||a|| * ||b||)
float cosine_scalar(const float* a, const float* b, std::size_t dim);
float cosine_avx2(const float* a, const float* b, std::size_t dim);
float cosine_avx512(const float* a, const float* b, std::size_t dim);

// ---------------------------------------------------------------------------
// Batch kernels: one query vs. N database vectors (row-major, dim floats/row)
// `results` must have space for n_vectors floats.
// Parallelized across rows with OpenMP; each row uses the SIMD kernel.
// ---------------------------------------------------------------------------
enum class Metric { DotProduct, L2Squared, Cosine };
enum class ImplKind { Scalar, AVX2, AVX512 };

void batch_compute(Metric metric, ImplKind impl,
                    const float* query,
                    const float* database, std::size_t n_vectors, std::size_t dim,
                    float* results,
                    int num_threads = 0 /* 0 = OpenMP default */);

// ---------------------------------------------------------------------------
// Inline definitions
// ---------------------------------------------------------------------------
inline float l2_scalar(const float* a, const float* b, std::size_t dim) {
    return std::sqrt(l2sq_scalar(a, b, dim));
}
inline float l2_avx2(const float* a, const float* b, std::size_t dim) {
    return std::sqrt(l2sq_avx2(a, b, dim));
}
inline float l2_avx512(const float* a, const float* b, std::size_t dim) {
    return std::sqrt(l2sq_avx512(a, b, dim));
}

}  // namespace vecops
