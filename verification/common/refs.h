#pragma once
// CPU reference implementations. Plain C++ (no CUDA), so they can also be
// compiled and sanity-checked on a machine without a GPU.
// Each one mirrors the GPU kernel's semantics, including operation ORDER,
// so float results can be compared bit-for-bit.
#include <cstddef>

// ArrayAddition: C[i] = B[i] + A[i]
inline void ref_vec_add(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) c[i] = b[i] + a[i];
}

// ImageBlur: 3x3 box blur on an interleaved 3-channel float image
// (index = 3*(row*width+col)+channel). At the borders the window shrinks and
// the sum is divided by the number of in-bounds pixels (not always 9).
// Summation order matches the kernel: rows outer, columns inner.
inline void ref_blur(const float* in, float* out, int height, int width) {
  for (int X = 0; X < height; ++X) {
    for (int Y = 0; Y < width; ++Y) {
      float s0 = 0, s1 = 0, s2 = 0;
      int count = 0;
      for (int i = X - 1; i <= X + 1; ++i) {
        for (int j = Y - 1; j <= Y + 1; ++j) {
          if (i >= 0 && j >= 0 && i < height && j < width) {
            size_t p = 3 * ((size_t)i * width + j);
            s0 += in[p];
            s1 += in[p + 1];
            s2 += in[p + 2];
            count++;
          }
        }
      }
      size_t o = 3 * ((size_t)X * width + Y);
      out[o]     = s0 / count;
      out[o + 1] = s1 / count;
      out[o + 2] = s2 / count;
    }
  }
}
