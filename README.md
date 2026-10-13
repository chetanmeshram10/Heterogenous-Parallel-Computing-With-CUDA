# Parallel Computing using CUDA, with a Verification Layer

CUDA C/C++ implementations of common parallel algorithms, extended with a **verification
layer**: CPU reference tests plus a Python harness that builds each tested kernel under
different `nvcc` settings and checks that every build produces the same, correct output
(a small-scale version of compiler differential testing).

> Based on [vilas897/Heterogenous-Parallel-Computing-with-CUDA](https://github.com/vilas897/Heterogenous-Parallel-Computing-with-CUDA).
> The kernels and the original code are the upstream author's. **My addition** is everything
> in `verification/`, plus small changes to the kernels needed to test them (see
> [What I changed](#what-i-changed-in-the-original-code)).

## Why this exists

A kernel that runs without crashing isn't necessarily correct, and one that is correct
at `-O0` may not be at `-O3`. This project answers two questions for each kernel:

1. Does the GPU output match a trusted CPU implementation?
2. Does it keep matching when compiled in different ways (optimisation levels, FMA
   contraction, different code-generation paths)?

## How it works

1. **CPU reference** (`verification/common/refs.h`): plain C++ that mirrors the GPU kernel's
   semantics, including the *order* of floating-point operations. That lets results be
   compared **bit-for-bit** instead of with a tolerance.
2. **Test** (`verification/tests/test_*.cu`): runs the kernel on deterministic inputs and
   edge cases (sizes that aren't multiples of the block size, 1x1 images, non-square shapes)
   and compares to the reference.
3. **Matrix runner** (`verification/run_matrix.py`): builds every test under the
   configurations below, runs it on the GPU, and checks each output against both the CPU
   reference and the unoptimised baseline (`-G`).

| Configuration | nvcc flags | What it exercises |
|---|---|---|
| `debug-G` (baseline) | `-G -O0` | device optimisations off |
| `O0` / `O1` / `O2` / `O3` | `-On -Xptxas -On` | host and device optimisation levels |
| `O3-nofma` | `--fmad=false` | no fused multiply-add contraction |
| `O3-ptx75` | `-gencode arch=compute_75,code=compute_75` | PTX JIT-compiled by the driver |
| `O3-fastmath` | `-use_fast_math` | expected to change float results (reported as a warning) |
| `build-sm_80`, `build-sm_90` | `-arch=sm_XX` | compile-only; the test GPU can't run them |

Note that `-O` only affects *host* code. Device-code optimisation is controlled by
`-Xptxas -O<n>`, and `-G` turns it off, so the matrix varies both.

## Results

Run on a **Tesla T4 (sm_75), nvcc 13.0**. Full log: [`verification/run_log.txt`](verification/run_log.txt).

<!-- VERIFY:START -->
_nvcc 13.0 · Tesla T4 (sm_75) · 2026-10-13_

| Kernel | debug-G | O0 | O1 | O2 | O3 | O3-nofma | O3-ptx75 | O3-fastmath | build-sm_80 | build-sm_90 |
|---|---|---|---|---|---|---|---|---|---|---|
| ArrayAddition | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🔨 | 🔨 |
| ImageBlur | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | 🔨 | 🔨 |

✅ matches CPU reference and baseline · ⚠️ lenient config differs (expected) · ❌ cpu = GPU differs from CPU · ❌ diff = differs from baseline · 🔧❌ build failed · 💥 crashed · 🔨 built only (not run)
<!-- VERIFY:END -->

The only deviation is ImageBlur under `-use_fast_math`: the kernel ends in a division, and
fast-math replaces IEEE division with a faster approximation. ArrayAddition (a plain add)
is unaffected.

On CUDA 13 the oldest supported target is `compute_75`, so the PTX column compares the
driver's JIT path against the normal ahead-of-time build on the same architecture rather
than an older GPU generation.

## Findings

Writing the tests exposed issues in the original code (details and evidence in
[`verification/FINDINGS.md`](verification/FINDINGS.md)):

- **ImageBlur: no bounds check (reproduced on the GPU).** Threads outside the image write
  past the end of the output buffer. `compute-sanitizer` reports
  `Invalid __global__ write of size 4 bytes` at `kernel.cuh:35`
  ([`sanitizer_original.txt`](verification/sanitizer_original.txt)). Fixed with a one-line guard;
  build with `-DBLUR_NO_GUARD` to reproduce the original behaviour.
- **ImageBlur: swapped launch arguments** (`<<<block, grid>>>` instead of `<<<grid, block>>>`).
  Found by reading the code; it works by accident for small images and would fail for
  images around 1024x1024 and larger. Fixed; not run on a GPU because the file needs the
  course's `wb.h`.
- **ImageBlur: possible buffer-size mismatch** (3 channels indexed, 1 channel allocated).
  Flagged for checking against `wb.h`; not changed.

## Repository layout

```
ArrayAddition/        original code; kernel moved to kernel.cuh
ImageBlur/            original code; kernel moved to kernel.cuh (+ bounds guard)
...                   other original kernel folders (not yet covered by tests)
verification/
├── common/
│   ├── verify.h      comparison + reporting helper (GPU vs CPU, JSON result line)
│   └── refs.h        CPU reference implementations
├── tests/            one test_<Kernel>.cu per covered kernel
├── run_matrix.py     builds/runs every test under every configuration
├── FINDINGS.md       bugs found, with evidence
├── RESULTS.md        latest results table
└── run_log.txt       raw log of the latest run
```

## Running it

Requires an NVIDIA GPU, the CUDA toolkit (`nvcc`), Python 3 and NumPy. No local GPU?
[Google Colab](https://colab.research.google.com) works: choose **Runtime, Change runtime type,
T4 GPU**, then:

```python
!git clone https://github.com/chetanmeshram10/Heterogenous-Parallel-Computing-with-CUDA.git
%cd Heterogenous-Parallel-Computing-with-CUDA
!python3 verification/run_matrix.py --compile-only sm_80,sm_90 --update-readme
```

Useful options: `--tests ImageBlur` (run one kernel), `--no-run --cc 75` (build only, no GPU
needed), `--ccbin g++-12` (pick the host compiler).

To reproduce the ImageBlur bug on a GPU:

```bash
nvcc -G -DBLUR_NO_GUARD -DNO_PAD -I verification -o /tmp/ng verification/tests/test_ImageBlur.cu
compute-sanitizer /tmp/ng /tmp/o.bin
```

## Adding a kernel

1. Move the `__global__` function into `<Kernel>/kernel.cuh` and `#include` it from the original `.cu`.
2. Add a CPU reference to `verification/common/refs.h`.
3. Write `verification/tests/test_<Kernel>.cu` (copy `test_ArrayAddition.cu`), using a
   `Verifier<T>` with tolerance 0 for integer or order-identical float kernels, or a small
   `abs_tol`/`rel_tol` where float summation order legitimately differs.
4. Run `run_matrix.py`.

## What I changed in the original code

- `ArrayAddition/ArraySum.cu`: kernel moved to `kernel.cuh`; it now takes the element count
  as a parameter instead of the `NUM` macro.
- `ImageBlur/template.cu`: kernel moved to `kernel.cuh`; launch arguments corrected.
- `ImageBlur/kernel.cuh`: added the bounds-check guard.

## Limitations

- Currently covers **2 kernels** (ArrayAddition, ImageBlur); the others are not yet tested.
- Results come from one GPU (Tesla T4) and one toolkit version; `sm_80`/`sm_90` are
  compile-checked only.
- This is differential testing of hand-written kernels, not a compiler fuzzer. It checks
  that known kernels behave consistently across flags, not that the compiler is correct
  in general.

## Original algorithms (upstream)

- Sum of elements in an array using parallel reduction (normal and Thrust)
- Addition of 2 arrays or matrices (normal and Thrust)
- Gaussian image blurring
- RGB to grayscale conversion
- Matrix multiplication (normal and tiled)
- Histogram of an ASCII string
- Histogram of an integer array (normal and Thrust)
- Image convolution
- 7-point stencil
- Inclusive prefix scan on a 1D array (normal and Thrust)

Each upstream folder has its own `README.md` with run instructions. Original code credit:
[vilas897](https://github.com/vilas897).
