# SIMD-Accelerated Vector Similarity Engine

A small C++ library that computes the three distance metrics used by every
RAG pipeline / vector database (**cosine similarity**, **L2 distance**, **dot
product**) over 768-dimensional embeddings, with hand-written **AVX2** and
**AVX-512** kernels benchmarked against a scalar baseline, and OpenMP
multi-threading across the batch dimension.

## Why this exists

Vector databases (FAISS, Milvus, ChromaDB's HNSW backend, pgvector) spend
the overwhelming majority of query time in exactly this inner loop: given a
query embedding, compute a similarity score against N candidate embeddings.
This project isolates that kernel and shows, concretely, where the speedup
comes from — instruction-level parallelism (SIMD lanes), FMA fusion, and
thread-level parallelism (OpenMP) — and how to measure it honestly.

## Layout

```
vector-simd-engine/
├── CMakeLists.txt
├── include/vector_ops.hpp      # public API
├── src/vector_ops.cpp          # scalar + AVX2 + AVX-512 kernels
├── benchmark/benchmark_main.cpp
└── tests/test_correctness.cpp
```

## Design

* **Single source file, multiple ISAs.** Instead of separate `.cpp` files
  compiled with different `-march` flags, each SIMD kernel is annotated with
  GCC/Clang's `__attribute__((target("avx2,fma")))` /
  `__attribute__((target("avx512f")))`. This is the same technique FAISS and
  hnswlib use for runtime dispatch: the binary stays portable (baseline
  build has no `-mavx2`), while `detect_cpu_features()` does a runtime
  `cpuid` check and callers pick the fastest kernel their CPU actually
  supports.
* **Honest scalar baseline.** The scalar kernels are compiled with
  `optimize("no-tree-vectorize", "no-tree-slp-vectorize")` so GCC's
  auto-vectorizer can't quietly turn them into SIMD code under `-O3` — the
  comparison is a *true* scalar loop vs. hand-written intrinsics, not
  "compiler auto-vectorized" vs. "compiler auto-vectorized differently."
* **Metrics implemented:**
  * `dot_*` — dot product, `sum(a[i]*b[i])`
  * `l2sq_*` — squared L2 distance (no `sqrt`; monotonic with true L2 and
    what every ANN index actually compares internally)
  * `l2_*` — true L2 distance (`sqrt` of the above, inline wrapper)
  * `cosine_*` — cosine similarity, computing dot product and both norms in
    a single pass over the data
* **AVX2 path:** 8×float32 per `__m256` register, FMA-fused
  multiply-accumulate, horizontal reduction via shuffle/add, scalar tail
  loop for `dim % 8 != 0`.
* **AVX-512 path:** 16×float32 per `__m512` register, `_mm512_reduce_add_ps`
  for reduction, same tail-loop handling.
* **Batch API:** `batch_compute()` takes one query against N database rows
  and parallelizes across rows with `#pragma omp parallel for`, so the
  thread-level and instruction-level parallelism compose (OpenMP threads
  each running the SIMD kernel).
* **Correctness:** `tests/test_correctness.cpp` checks AVX2/AVX-512 against
  scalar across dims that are exact multiples of 8/16 *and* awkward
  non-multiples (1, 7, 17, 763, 770) to exercise the tail loop, plus a
  1M-scale... (2000-row, kept small for CI speed) batch check.

## Build

Requires a C++17 compiler (**GCC or Clang** — MSVC is not supported, see
note below), CMake ≥ 3.16, and OpenMP.

### Linux / macOS

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
./unit_tests     # correctness
./benchmark      # 1M x 768-dim benchmark (override: ./benchmark <n_vectors> <dim>)
```

OpenMP: `libgomp` ships with GCC on Linux (nothing extra to install). On
macOS, Apple Clang doesn't bundle OpenMP, so install it first:
```bash
brew install libomp
```

### Windows

Windows doesn't ship `make`, and the default Visual Studio toolchain
(MSVC) doesn't support the GCC/Clang-specific intrinsics dispatch this
project uses (`__attribute__((target(...)))`, `__builtin_cpu_supports`,
`<cpuid.h>`). So you need a real GCC toolchain — **MinGW-w64**, easiest
installed via [MSYS2](https://www.msys2.org/):

1. Install MSYS2:
   ```powershell
   winget install -e --id MSYS2.MSYS2
   ```
2. Open the **"MSYS2 UCRT64"** shell (Start menu — not the plain MSYS2
   shell) and install the compiler + build tools:
   ```bash
   pacman -Syu
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-make
   ```
3. Add `C:\msys64\ucrt64\bin` to your **User PATH** (System Properties →
   Environment Variables → Path → New), then open a **fresh** terminal
   (PowerShell or VS Code) so it picks up the change. Verify with:
   ```powershell
   g++ --version
   ```
4. Build using the MinGW Makefiles generator (the default Visual Studio
   generator will pick MSVC and fail):
   ```powershell
   mkdir build
   cd build
   cmake .. -G "MinGW Makefiles"
   cmake --build .
   .\unit_tests.exe
   .\benchmark.exe
   ```

OpenMP ships with the MinGW-w64 GCC package above (`libgomp`), so no
separate install is needed.

**Portability notes baked into the code so this works cross-platform:**
* Aligned allocation uses `_aligned_malloc`/`_aligned_free` on Windows
  (`#if defined(_WIN32)`) and `posix_memalign`/`free` elsewhere —
  `posix_memalign` doesn't exist on Windows at all (MinGW or MSVC).
* `detect_cpu_features()`, `<cpuid.h>`, and `<immintrin.h>` are genuine
  GCC/MinGW functionality (not POSIX-only), so they build the same way on
  Linux, macOS, and MinGW-w64 without `#ifdef`s.

## Benchmark results

Hardware: **Intel Core Ultra 7 155H** (14 cores / 22 threads — 6 P-cores +
8 E-cores + 2 low-power E-cores, "Meteor Lake"), 32 GB RAM, Windows 11.

```
=== SIMD-Accelerated Vector Similarity Engine ===
Vectors: 1000000   Dim: 768   (2929.7 MB dataset)
CPU features: AVX2=yes FMA=yes AVX-512F=no
OpenMP max threads: 22

--- Dot Product ---
  scalar (1 thread)                537.41 ms          1860770 vec/s     1.00x
  AVX2+FMA (1 thread)              158.16 ms          6322719 vec/s     3.40x
  AVX2+FMA (22 threads)             40.32 ms         24800234 vec/s    13.33x
  max |scalar - AVX2|   = 5.341e-05

--- L2 Squared Distance ---
  scalar (1 thread)                625.53 ms          1598648 vec/s     1.00x
  AVX2+FMA (1 thread)              167.60 ms          5966623 vec/s     3.73x
  AVX2+FMA (22 threads)             38.87 ms         25727576 vec/s    16.09x
  max |scalar - AVX2|   = 1.099e-03

--- Cosine Similarity ---
  scalar (1 thread)                787.44 ms          1269945 vec/s     1.00x
  AVX2+FMA (1 thread)              168.00 ms          5952530 vec/s     4.69x
  AVX2+FMA (22 threads)             40.40 ms         24750760 vec/s    19.49x
  max |scalar - AVX2|   = 2.384e-07
```

Max absolute error vs. scalar stays in the `1e-4`–`1e-7` range across all
three metrics (float reassociation, not a bug — SIMD sums 8/16 partial
lanes before the final horizontal reduction, so it doesn't accumulate in
the same order as the scalar loop).

**No AVX-512 row above — and why:** the Core Ultra 7 155H is a *hybrid*
chip (performance cores + efficiency cores on the same die, like Intel's
12th/13th/14th-gen desktop parts and all Meteor Lake/Arrow Lake mobile
chips). Intel disabled AVX-512 across its entire consumer/mobile lineup
starting with that hybrid generation: the E-cores don't implement AVX-512
in silicon, and the OS scheduler can migrate a thread from a P-core to an
E-core mid-execution, so Intel couldn't guarantee an AVX-512 instruction
wouldn't land on a core that can't execute it. Rather than risk that,
they fused the feature off entirely — even on the P-cores, which
physically could support it. AVX-512 now effectively only survives on
Xeon server/workstation parts and a handful of older non-hybrid desktop
CPUs (10th-gen Comet Lake and earlier).

`detect_cpu_features()` does a real runtime `cpuid` check rather than
assuming anything at compile time, so it correctly reports
`avx512f = false` here, and the benchmark/tests skip those code paths
entirely (`if (cpu.avx512f) { ... }`) rather than risking an
illegal-instruction crash. Practically, this costs almost nothing: as the
scaling numbers below show, this workload is memory-bandwidth-bound, and
AVX-512 typically only adds ~10-15% over AVX2 on bandwidth-bound kernels
like this one (doubling the SIMD register width doesn't help once you're
already waiting on RAM) — so the AVX2 results above already capture the
large majority of the achievable speedup on this CPU.

**Why 22 threads gives ~4x, not ~22x, over single-threaded AVX2:** going
from 1 → 22 threads on the AVX2 kernel took Dot Product from 158 ms → 40 ms
(≈3.9x), L2 from 168 ms → 39 ms (≈4.3x), and Cosine from 168 ms → 40 ms
(≈4.2x) — real speedup, but nowhere near linear with core count. At 768
floats (3 KB) per vector × 1M vectors (~2.9 GB total), this workload is
**memory-bandwidth-bound**: once enough threads are reading from RAM in
parallel, the memory bus itself becomes the bottleneck (typically saturating
around 4-8 cores on a single-socket system), and adding more threads beyond
that point mostly means more cores idling on cache misses rather than more
useful work per second. This is the same reason AVX-512 doesn't scale much
past AVX2 above — both are hitting the same wall from different angles
(wider SIMD lanes vs. more threads), and it's the standard justification
production vector-DB engines give for leaning on quantization (int8/PQ) to
shrink bytes-moved-per-comparison instead of just adding more parallelism.

## Future work

* Batch a **matrix of queries** against the database (all-pairs), not just
  one query — turns this into a GEMM-like kernel and lets you compare
  against BLAS.
* Add int8 quantized kernels (`_mm256_maddubs_epi16` / VNNI) — this is what
  actually gets used in production ANN indexes to cut memory bandwidth 4x.
* Add a simple flat-index top-K search (partial sort over `results`) to
  turn this into an actual (brute-force) nearest-neighbor search engine.
* Compare against `-O3 -march=native` **auto-vectorized** scalar code (drop
  the `no-tree-vectorize` attribute) to see how close the compiler gets on
  its own vs. hand intrinsics.