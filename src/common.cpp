// common.cpp
// Aligned allocation, runtime CPU feature detection, and the OpenMP batch
// dispatcher. Compiled with NO special -march/-arch flags, so it must not
// call any AVX/AVX-512 intrinsics directly — it only detects what's
// available and calls through function pointers into the kernel files that
// *are* built with the right per-file flags (see CMakeLists.txt).

#include "vector_ops.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vecops {

// ---------------------------------------------------------------------------
// Aligned allocation
// ---------------------------------------------------------------------------
void AlignedDeleter::operator()(float* p) const noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

AlignedBuffer make_aligned(std::size_t count) {
    constexpr std::size_t kAlign = 64;
    std::size_t bytes = count * sizeof(float);
    if (bytes == 0) bytes = kAlign;
    void* raw = nullptr;
#if defined(_MSC_VER)
    raw = _aligned_malloc(bytes, kAlign);
    if (!raw) throw std::bad_alloc();
#else
    if (posix_memalign(&raw, kAlign, bytes) != 0) throw std::bad_alloc();
#endif
    std::memset(raw, 0, bytes);
    return AlignedBuffer(static_cast<float*>(raw));
}

// ---------------------------------------------------------------------------
// CPU feature detection (CPUID + XGETBV, so it also confirms the OS has
// enabled the relevant register state, not just that the silicon has it).
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define VECOPS_X86 1
#endif

#ifdef VECOPS_X86
namespace {

void cpuid(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}

std::uint64_t xgetbv(unsigned int index) {
#if defined(_MSC_VER)
    return _xgetbv(index);
#else
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

}  // namespace
#endif  // VECOPS_X86

CpuFeatures detect_cpu_features() {
    CpuFeatures f;
#ifdef VECOPS_X86
    int regs1[4] = {0, 0, 0, 0};
    cpuid(1, 0, regs1);
    bool osxsave = (regs1[2] >> 27) & 1;
    bool fma_bit = (regs1[2] >> 12) & 1;
    bool avx_bit = (regs1[2] >> 28) & 1;

    bool avx_os_ok = false;
    bool avx512_os_ok = false;
    if (osxsave) {
        std::uint64_t xcr0 = xgetbv(0);
        avx_os_ok = (xcr0 & 0x6) == 0x6;      // XMM (bit1) + YMM (bit2) state
        avx512_os_ok = (xcr0 & 0xE6) == 0xE6;  // + opmask/ZMM-hi state (bits 5-7)
    }

    int regs7[4] = {0, 0, 0, 0};
    cpuid(7, 0, regs7);
    bool avx2_bit = (regs7[1] >> 5) & 1;
    bool avx512f_bit = (regs7[1] >> 16) & 1;

    f.avx2 = avx_bit && avx2_bit && avx_os_ok;
    f.fma = fma_bit && avx_os_ok;
    f.avx512f = avx512f_bit && avx512_os_ok;
#endif
    return f;
}

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
    for (long long row = 0; row < static_cast<long long>(n_vectors); ++row) {
        const float* vec = database + static_cast<std::size_t>(row) * dim;
        results[row] = kernel(query, vec, dim);
    }
}

}  // namespace vecops
