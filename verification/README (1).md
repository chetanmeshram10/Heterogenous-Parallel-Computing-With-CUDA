# Parallel Computing using CUDA

This repository consists of implementations of various parallel algorithms using `CUDA`.

Each folder contains a `README.md` file containing the instructions to run the codes

#### List of algorithms implemented

- Sum of elements in an array using parallel reduction (Normal and Thrust Implementation)
- Addition of 2 arrays or matrices (Normal and Thrust Implementation)
- Gaussian Image Blurring Algorithm
- RGB to Gray Scale Image conversion
- Matrix Multiplication (Both Normal and Tiled Approach)
- Histogram of ASCII Character String
- Histogram of Integer Array (Normal and Thrust Implementation)
- Image Convolution
- 7 point Stencil
- Inclusive Prefix Scan on a 1D array (Normal and Thrust Implementation) 


> Based on [vilas897/Heterogenous-Parallel-Computing-with-CUDA](https://github.com/vilas897/Heterogenous-Parallel-Computing-with-CUDA).
> **My addition:** `verification/`, a CPU-reference test for each kernel plus a Python
> harness that builds every kernel under different `nvcc` flags and checks that all
> outputs agree (compiler differential testing). See `verification/FINDINGS.md`
> for bugs found in the original kernels.

## Verification

Run: `python3 verification/run_matrix.py --update-readme`

<!-- VERIFY:START -->
_nvcc 13.0 · Tesla T4 (sm_75) · 2026-10-13_

| Kernel | debug-G | O0 | O1 | O2 | O3 | O3-nofma | O3-ptx75 | O3-fastmath | build-sm_80 | build-sm_90 |
|---|---|---|---|---|---|---|---|---|---|---|
| ArrayAddition | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🔨 | 🔨 |
| ImageBlur | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | 🔨 | 🔨 |

✅ matches CPU reference and baseline · ⚠️ lenient config differs (expected) · ❌ cpu = GPU differs from CPU · ❌ diff = differs from baseline · 🔧❌ build failed · 💥 crashed · 🔨 built only (not run)
<!-- VERIFY:END -->