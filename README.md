# CUDA Matrix Multiplication — GPU Acceleration

This repo implements matrix multiplication (SGEMM: `C = α·A·B + β·C`) in CUDA across five progressively optimized kernels — from a naive baseline to 2D block tiling — and benchmarks each against the others to show where the speedup actually comes from.

| Kernel | Speedup vs. naive |
|---|---|
| Naive global memory | 1x (baseline) |
| Global memory coalescing | ~2-4x |
| Shared memory cache blocking | ~10-20x |
| Same-size tiled (16x16) | ~15-25x |
| General 2D block tiling | ~20-30x+ |

(Relative speedups depend heavily on your GPU — see [Benchmarking](#benchmarking) for how to generate your own numbers. The project description reports **over 1000x speedup versus a CPU implementation**.)

---

## Table of Contents

1. [How GPU Computing Works (CUDA Basics)](#how-gpu-computing-works-cuda-basics)
2. [Visualizing Threads, Blocks, and Grids](#visualizing-threads-blocks-and-grids)
3. [The Memory Hierarchy](#the-memory-hierarchy)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [Building and Running](#building-and-running)
6. [Benchmarking](#benchmarking)
7. [Key Optimization Concepts](#key-optimization-concepts)

---

## How GPU Computing Works (CUDA Basics)

A CPU has a handful of powerful cores optimized for doing one thing very fast, one after another (with some parallelism via multiple cores/threads). A GPU flips that trade-off: it has **thousands of small, simple cores** optimized for doing the *same* operation on *many* pieces of data at once. This is called **SIMT** (Single Instruction, Multiple Threads).

Matrix multiplication is a perfect fit for this model: computing each output element `C[i][j]` is an independent dot product of row `i` of `A` and column `j` of `B`. None of those dot products depend on each other, so instead of looping over every `(i, j)` pair one at a time like a CPU would, a GPU can compute **thousands of them simultaneously**.

### The CPU → GPU workflow (used in every file in this repo)

1. **Allocate memory on the GPU** (`cudaMalloc`) — the CPU ("host") and GPU ("device") have physically separate memory, so the GPU needs its own buffers.
2. **Copy input data from host to device** (`cudaMemcpy(..., cudaMemcpyHostToDevice)`) — matrices `A` and `B` are generated on the CPU and shipped over the PCIe bus to the GPU.
3. **Launch a kernel** (`kernel<<<gridDim, blockDim>>>(...)`) — this is a function that runs on the GPU, but you don't call it once — you launch it across a whole grid of threads, and every thread runs the same kernel code on its own slice of the data.
4. **Synchronize** (`cudaDeviceSynchronize()`) — kernel launches are asynchronous from the CPU's perspective, so this blocks until the GPU finishes.
5. **Copy results back** (`cudaMemcpyDeviceToHost`).
6. **Free memory** (`cudaFree`, `free`).

This host/device round-trip is why GPU acceleration is worth it only when there's *enough work* to hide the cost of steps 1–2 and 5 — small matrices might actually be slower on a GPU than a CPU because of this overhead.

---

## Visualizing Threads, Blocks, and Grids

CUDA organizes parallel work into a 3-level hierarchy. Every kernel launch in this repo uses the `<<<gridDim, blockDim>>>` syntax to define this hierarchy:

```
GRID  (the whole problem — e.g. the entire output matrix C)
 └── BLOCK  (a tile of the problem, e.g. a 32x32 chunk of C)
      └── THREAD  (one unit of work, e.g. computing a single C[row][col])
```

### Picture it like a spreadsheet

Imagine matrix `C` (size `M x N`) laid out on a wall. You chop it into square tiles of size `BLOCK_SIZE x BLOCK_SIZE` (e.g. 32x32). Each **tile is a block**. Inside each block, every **cell is a thread**, and that thread is responsible for computing exactly one output element.

```
                    N columns of C
        ┌───────────┬───────────┬───────────┐
        │  Block     │  Block     │  Block     │
        │  (0,0)     │  (1,0)     │  (2,0)     │   ← gridDim.x = 3 blocks wide
        │ 32x32      │ 32x32      │ 32x32      │
   M    │  threads   │  threads   │  threads   │
 rows   ├───────────┼───────────┼───────────┤
of C    │  Block     │  Block     │  Block     │
        │  (0,1)     │  (1,1)     │  (2,1)     │   ← gridDim.y = 2 blocks tall
        │ 32x32      │ 32x32      │ 32x32      │
        └───────────┴───────────┴───────────┘

Zoomed into one Block (0,0):
        ┌────┬────┬────┬─── ... ──┬────┐
        │T00 │T10 │T20 │          │T31,0│   ← threadIdx.x = 0..31
        ├────┼────┼────┼── ... ──┼────┤
        │T01 │T11 │T21 │          │T31,1│
        ├────┼────┼────┼── ... ──┼────┤
        │ .. │ .. │ .. │   ...    │ .. │
        └────┴────┴────┴── ... ──┴────┘
        threadIdx.y = 0..31
```

Every thread can figure out exactly which element of `C` it owns using its coordinates:

```cpp
int row    = blockIdx.y * blockDim.y + threadIdx.y;
int column = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx` — which block am I in? (position in the grid)
- `blockDim` — how big is each block? (e.g. 32x32 threads)
- `threadIdx` — where am I inside my block?

This single line is the mental model for *every* kernel in this repo — it's how a flat launch of thousands of threads maps back onto 2D matrix coordinates.

### Warps — the real unit of execution

Threads inside a block are actually scheduled in groups of **32 called a warp**. All 32 threads in a warp execute the same instruction in lockstep. This detail matters a lot for performance — it's the entire reason the "memory coalescing" kernel in this repo exists (see below): if the 32 threads in a warp access scattered memory addresses, the hardware has to issue multiple slow memory transactions instead of one fast one.

### Visualizing it yourself

- **Nsight Compute / Nsight Systems** (NVIDIA's free profilers) will show you real occupancy, warp execution, and memory throughput for these kernels.
- `compute-sanitizer` can catch out-of-bounds thread accesses.
- Conceptually, the ASCII diagram above *is* the visualization — draw the output matrix, chop it into blocks, and shrink each block into a grid of threads.

---

## The Memory Hierarchy

This is the single most important concept for understanding why the kernels in this repo get progressively faster. GPU memory is tiered by size and speed:

```
Registers        (per-thread,  fastest,   tiny)
Shared Memory    (per-block,   very fast, ~48-228 KB)
L2 Cache         (per-GPU,     fast)
Global Memory    (per-GPU,     slow,      GBs of capacity)
```

Every kernel needs data from `A` and `B`, which live in slow **global memory**. The optimization story of this repo is entirely about **moving data closer to the compute** (into shared memory / registers) and **minimizing how many times slow global memory gets touched**.

---

## File-by-File Breakdown

All five files are standalone, self-contained `.cu` programs. Each one: generates random `M x K` and `K x N` matrices, times 3 warm-up kernel launches, then benchmarks 20 timed launches and prints the average runtime. This makes them directly comparable to each other.

### `SGEMM.cu` — Naive Baseline

The simplest possible GPU implementation, and the reference point every other file is measured against.

- Each thread computes exactly one output element `C[row][col]` by looping over the full `k` dimension and doing a straight dot product, reading directly from global memory every single time.
- **Why it's slow:** every one of the `k` multiply-adds per thread issues a fresh global memory read for both `A` and `B` — no reuse, no caching. With 32x32 = 1024 threads per block all hammering global memory independently, the memory bus becomes the bottleneck, not the math.
- **Benefit of this file:** it's the control group. Nothing here is wrong, but it establishes the "before" picture so the other four kernels' improvements are measurable and explainable.

### `Global_Memory_Coalescing_MM.cu` — Memory Coalescing

Same basic per-thread-one-output algorithm, but the way threads are mapped to memory addresses changes:

```cpp
const int row    = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
const int column = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
```

- Instead of using a 2D `threadIdx.(x,y)`, this flattens thread indexing so that **consecutive threads in a warp read consecutive addresses in memory** (adjacent `column` values for `B`). This lets the GPU merge, or "coalesce," many individual thread memory requests into one wide, efficient transaction instead of dozens of small scattered ones.
- Also introduces the general SGEMM signature `C = α·(A·B) + β·C`, matching how real BLAS libraries (like cuBLAS) define the operation.
- **Benefit:** free performance from *only* changing memory access patterns — no algorithmic change, no extra memory used. This demonstrates that *how* you read memory matters as much as *what* you compute.

### `Shared_Memory_Cache_Blocking.cu` — Shared Memory Caching

This is where the algorithm itself changes, not just the memory access pattern.

- Declares `__shared__ float shA[...]` and `shB[...]` — small, fast, on-chip memory shared by every thread in a block.
- Each block cooperatively loads a `BLOCK_SIZE x BLOCK_SIZE` tile of `A` and `B` from slow global memory into fast shared memory **once**, calls `__syncthreads()` to make sure every thread has finished loading before anyone starts computing, then every thread in the block reuses that cached tile to accumulate a partial dot product. The loop slides across `K` in `BLOCK_SIZE`-sized chunks, repeating load → sync → compute → sync.
- **Why it's much faster:** each value loaded from global memory is now reused `BLOCK_SIZE` times (by every thread in the row/column that needs it) instead of being re-fetched from global memory by every single thread that needs it. This is the single biggest lever in GEMM optimization — cutting global memory traffic by roughly a factor of `BLOCK_SIZE`.
- **Benefit:** dramatically reduces global memory bandwidth pressure, which is almost always the real bottleneck in GPU compute — not arithmetic throughput.

### `Same_Size_Tiled_Matrix_Multiplication.cu` — 2D Tiling (Square Matrices)

Builds on shared-memory caching with a cleaner, more general tiling loop, but restricted to square `N x N` matrices.

- Uses explicit 2D shared memory arrays (`shA[TILE_SIZE][TILE_SIZE]`) rather than flattened 1D arrays, which is more readable and maps directly onto `threadIdx.x` / `threadIdx.y`.
- Adds **boundary checking** (`if (row < N) ... else shA[ty][tx] = 0.0f`) so tiles that don't perfectly divide the matrix size still work safely — the earlier files assume dimensions divide evenly by `BLOCK_SIZE`.
- **Benefit:** demonstrates the tiling pattern in its clearest, most textbook form — a good file to read first if you want to understand the *shape* of the algorithm before the more general/optimized versions.

### `General_2D_BlockTiled_MM.cu` — Generalized 2D Block Tiling

The most complete and production-realistic kernel in the repo.

- Identical tiling structure to the file above, but generalized to **arbitrary, non-square dimensions** (`M x K` times `K x N`), using `ceil((float)k / TILE_SIZE)` to correctly handle any matrix size, not just ones that divide evenly.
- Adds boundary checks on *both* the load step and the final write (`if ((row < m) && (column < n))`), so it's safe for real-world matrix shapes where `M`, `K`, `N` aren't multiples of the tile size.
- **Benefit:** this is the kernel you'd actually adapt for a real application — it keeps all the shared-memory reuse benefits above while removing the "must be square / must divide evenly" restriction, at the cost of a few extra branch checks per thread.

### Progression at a glance

| File | Memory pattern | Shared memory? | Handles non-square? |
|---|---|---|---|
| `SGEMM.cu` | Naive, uncoalesced | No | Yes (M,K,N params) |
| `Global_Memory_Coalescing_MM.cu` | Coalesced | No | Yes (M,K,N params) |
| `Shared_Memory_Cache_Blocking.cu` | Coalesced + cached | Yes (1D) | Yes (M,K,N params) |
| `Same_Size_Tiled_Matrix_Multiplication.cu` | Coalesced + cached | Yes (2D) | No (N x N only) |
| `General_2D_BlockTiled_MM.cu` | Coalesced + cached | Yes (2D) | Yes, with bounds checks |

---

## Building and Running

Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and an NVIDIA GPU.

```bash
# Compile any individual file with nvcc
nvcc -O3 SGEMM.cu -o sgemm
nvcc -O3 Global_Memory_Coalescing_MM.cu -o coalesced
nvcc -O3 Shared_Memory_Cache_Blocking.cu -o shared_mem -lcublas
nvcc -O3 Same_Size_Tiled_Matrix_Multiplication.cu -o tiled_square
nvcc -O3 General_2D_BlockTiled_MM.cu -o tiled_general

# Run any of them
./sgemm
./coalesced
./shared_mem
./tiled_square
./tiled_general
```

> `Shared_Memory_Cache_Blocking.cu` includes `<cublas_v2.h>`, so it needs `-lcublas` at link time even though the kernel itself doesn't call cuBLAS directly.

Each program prints something like:

```
Performing warm-up runs...
Benchmarking GPU implementation...
GPU average time: 143.827000 microseconds
```

## Benchmarking

Each file already contains its own benchmark harness (3 warm-up launches + 20 timed launches, averaged). To compare kernels head-to-head:

1. Build all five binaries as shown above.
2. Run each and record the printed average time.
3. Because `M`, `K`, `N`, and `BLOCK_SIZE`/`TILE_SIZE` are `#define`d at the top of each file, keep them consistent across files if you want an apples-to-apples comparison.
4. For deeper profiling (occupancy, memory throughput, warp efficiency), run any binary through:
   ```bash
   ncu ./tiled_general        # Nsight Compute
   nsys profile ./tiled_general   # Nsight Systems
   ```

## Key Optimization Concepts

A quick-reference glossary for the ideas used across these files:

- **Coalesced memory access** — arranging thread-to-data mapping so consecutive threads read consecutive memory addresses, letting the hardware combine many small reads into one large, efficient transaction.
- **Shared memory tiling** — cooperatively loading a small block of the input matrices into fast on-chip shared memory once, then having every thread in the block reuse it, instead of every thread re-reading from slow global memory.
- **`__syncthreads()`** — a barrier that forces every thread in a block to wait until all threads reach that point. Essential after loading into shared memory (so no thread starts computing with a half-filled tile) and after computing (so no thread overwrites shared memory that others still need).
- **Occupancy** — how many warps are actively resident on a streaming multiprocessor at once. Higher occupancy generally means better latency hiding, but using more shared memory or registers per thread can lower it — a trade-off implicit in choosing `BLOCK_SIZE`/`TILE_SIZE`.
- **Boundary checking** — guarding array accesses with `if (row < M && col < N)` so kernels don't read/write out of bounds when matrix dimensions aren't exact multiples of the block/tile size.
