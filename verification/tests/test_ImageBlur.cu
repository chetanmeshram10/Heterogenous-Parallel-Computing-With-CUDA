#include <limits>
#include "../common/verify.h"
#include "../common/refs.h"
#include "../../ImageBlur/kernel.cuh"

// Same arithmetic order on CPU and GPU, additions plus one IEEE division, so
// the results must match bit-for-bit (abs_tol = rel_tol = 0).
int main(int argc, char** argv) {
  const char* out_path = argc > 1 ? argv[1] : "out.bin";
  Verifier<float> v("ImageBlur", 0.0, 0.0);

  struct Dim { int h, w; };
  // Tiny, non-square, off-by-one around the 16x16 block, and a realistic size.
  const Dim dims[] = {{1, 1}, {1, 7}, {7, 1}, {3, 3}, {16, 16}, {17, 33},
                      {32, 32}, {33, 31}, {100, 257}, {480, 640}};
  const float kCanary = 12345.0f;
  const float kNaN = std::numeric_limits<float>::quiet_NaN();

  for (const Dim& d : dims) {
    const int h = d.h, w = d.w;
    const size_t n = (size_t)h * w * 3;
#ifdef NO_PAD
    const size_t pad = 0;   // exact-size buffer: lets compute-sanitizer see overruns
#else
    // Extra room after the output so an unguarded kernel's stray writes are
    // contained and can be detected via the canary values below.
    const size_t pad = (size_t)3 * 16 * (w + 16);
#endif

    auto in = make_random<float>(n, 7, 0.0, 1.0);
    std::vector<float> ref(n);
    ref_blur(in.data(), ref.data(), h, w);

    // Output starts as NaN (catches pixels the kernel never writes);
    // the padding starts as a canary (catches writes past the image).
    std::vector<float> init(n + pad, kNaN);
    for (size_t i = n; i < n + pad; ++i) init[i] = kCanary;

    float *din, *dout;
    CUDA_CHECK(cudaMalloc((void**)&din, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dout, (n + pad) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(din, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dout, init.data(), (n + pad) * sizeof(float), cudaMemcpyHostToDevice));

    // grid.x covers rows (the kernel's X = height), grid.y covers columns.
    dim3 block(16, 16, 1);
    dim3 grid((h + 15) / 16, (w + 15) / 16, 1);
    blur<<<grid, block>>>(din, dout, h, w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> back(n + pad);
    CUDA_CHECK(cudaMemcpy(back.data(), dout, (n + pad) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(din); cudaFree(dout);

    std::string tag = std::to_string(h) + "x" + std::to_string(w);
    v.check(std::vector<float>(back.begin(), back.begin() + n), ref, ("image " + tag).c_str());
    if (pad) {
      v.check(std::vector<float>(back.begin() + n, back.end()),
              std::vector<float>(pad, kCanary), ("canary " + tag).c_str());
    }
  }
  return v.finish(out_path);
}
