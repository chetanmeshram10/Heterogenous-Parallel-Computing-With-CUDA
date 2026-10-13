# Findings from the verification layer

Issues found in the original kernels while writing CPU references and tests.
Each entry says how strongly it has been confirmed.

## 1. ImageBlur: kernel has no bounds check (out-of-bounds writes)

`blur` computes `X`/`Y` from the thread index and writes
`output[3*(X*width+Y)+c]` without checking `X < height && Y < width`.
Any launch whose grid is rounded up past the image size has threads outside
the image. They write past the end of the buffer (rows beyond `height`), or,
for `Y >= width`, into the *next row's* pixels, racing with the legitimate
thread. Threads with `count == 0` also compute `0/0`.

* **Fix:** one guard line at the top of the kernel (`ImageBlur/kernel.cuh`).
  Build with `-DBLUR_NO_GUARD` to restore the original behaviour.
* **Evidence:** reproduced on a Tesla T4 (CUDA 13.0). `compute-sanitizer` reports
  `Invalid __global__ write of size 4 bytes` at `kernel.cuh:35` (the first output
  store) from the first test case, a 1x1 image with a 12-byte output buffer, with
  225 errors before the run aborts (`verification/sanitizer_original.txt`).
  With the guard in place the same check reports 0 errors
  (`verification/sanitizer_guarded.txt`).

Reproduce on a GPU:
```bash
# wrong pixels + canary overwrites reported by the test
nvcc -G -DBLUR_NO_GUARD -I verification -o /tmp/ng verification/tests/test_ImageBlur.cu && /tmp/ng /tmp/o.bin
# exact-size buffers so the sanitizer can see the overrun
nvcc -G -lineinfo -DBLUR_NO_GUARD -DNO_PAD -I verification -o /tmp/ng2 verification/tests/test_ImageBlur.cu
compute-sanitizer /tmp/ng2 /tmp/o.bin
```

## 2. ImageBlur: kernel launch arguments are swapped (`template.cu`)

The code launches `blur<<<local_size, global_size>>>`, but the order is
`<<<grid, block>>>`, so the 32x32 "local" size became the grid and the
computed size became the block. It works by accident for small images (it
still covers every pixel), but the block then has `(h/32+1)*(w/32+1)` threads,
which exceeds the 1024-thread limit for images of roughly 1024x1024 and up.
The launch then fails silently because the code never checks for errors.

* **Fix:** swapped the two arguments.
* **Evidence:** from reading the code. Not run on a GPU (the file needs the
  course's `wb.h`).

## 3. ImageBlur: possible buffer-size mismatch (not changed)

The kernel indexes the image as 3 interleaved channels (`3*(i*width+j)`), but
`template.cu` allocates `imageWidth * imageHeight * sizeof(float)`, i.e. one
channel, and its comment calls the image grayscale. Check `wbImage_new`'s
channel argument in `wb.h`: if the image really has 3 channels per pixel the
allocation and copies are 3x too small.

* **Evidence:** from reading the code only; depends on `wb.h`.

## 4. Minor: unchecked CUDA calls (ArraySum.cu)

`cudaMemcpy` results and the kernel launch are never checked, so failures are
silent. The tests use `CUDA_CHECK` and `cudaGetLastError()` everywhere.
