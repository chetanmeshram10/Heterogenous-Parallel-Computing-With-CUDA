#include "../common/verify.h"
#include "../common/refs.h"
#include "../../ArrayAddition/kernel.cuh"

int main(int argc, char** argv) {
  const char* out_path = argc > 1 ? argv[1] : "out.bin";
  // One IEEE float add is correctly rounded, so GPU and CPU must match exactly.
  Verifier<float> v("ArrayAddition", 0.0, 0.0);

  // 327133 is the size used in the original ArraySum.cu. Others straddle the
  // 1024-thread block size.
  const int sizes[] = {1, 31, 1023, 1024, 1025, 327133, (1 << 20) + 3};

  for (int n : sizes) {
    for (int regime = 0; regime < 2; ++regime) {
      // regime 0: large positives like the original's rand() values (up to 2^31)
      // regime 1: mixed-sign values, so cancellation is exercised
      double lo = regime ? -1000.0 : 0.0;
      double hi = regime ? 1000.0 : 2147483648.0;
      auto a = make_random<float>(n, 1 + regime, lo, hi);
      auto b = make_random<float>(n, 11 + regime, lo, hi);
      std::vector<float> ref(n), gpu(n);
      ref_vec_add(a.data(), b.data(), ref.data(), n);

      float *da, *db, *dc;
      CUDA_CHECK(cudaMalloc((void**)&da, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void**)&db, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void**)&dc, n * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

      add<<<(n + 1023) / 1024, 1024>>>(da, db, dc, n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(gpu.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
      cudaFree(da); cudaFree(db); cudaFree(dc);

      std::string label = "n=" + std::to_string(n) + ",regime=" + std::to_string(regime);
      v.check(gpu, ref, label.c_str());
    }
  }
  return v.finish(out_path);
}
