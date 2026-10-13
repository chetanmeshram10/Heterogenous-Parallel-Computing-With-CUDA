#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_),   \
              __FILE__, __LINE__);                                            \
      exit(2);                                                                \
    }                                                                         \
  } while (0)

template <typename T> const char* dtype_name() {
  if (std::is_same<T, float>::value) return "float32";
  if (std::is_same<T, double>::value) return "float64";
  if (std::is_same<T, int>::value) return "int32";
  if (std::is_same<T, unsigned int>::value) return "uint32";
  if (std::is_same<T, unsigned char>::value) return "uint8";
  return "unknown";
}

// Deterministic input. integral=true gives whole numbers (stored in T).
template <typename T>
std::vector<T> make_random(size_t n, uint32_t seed, double lo, double hi,
                           bool integral = false) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> d(lo, hi);
  std::vector<T> v(n);
  for (auto& x : v) { double r = d(rng); x = (T)(integral ? std::floor(r) : r); }
  return v;
}

// Collects GPU-vs-CPU comparisons across many test cases, then writes all GPU
// output to one binary file (for cross-compiler comparison) and prints one
// JSON line. Pass: |gpu-ref| <= abs_tol + rel_tol*|ref|. Both 0 means exact.
template <typename T>
class Verifier {
 public:
  Verifier(std::string kernel, double abs_tol = 0.0, double rel_tol = 0.0)
      : kernel_(std::move(kernel)), abs_tol_(abs_tol), rel_tol_(rel_tol) {}

  void check(const std::vector<T>& gpu, const std::vector<T>& ref,
             const char* label) {
    if (gpu.size() != ref.size()) { fprintf(stderr, "size mismatch\n"); exit(2); }
    cases_++;
    size_t bad_here = 0;
    for (size_t i = 0; i < gpu.size(); ++i) {
      double g = (double)gpu[i], r = (double)ref[i];
      bool ok;
      if (std::isnan(g) || std::isnan(r)) {
        ok = std::isnan(g) && std::isnan(r);
      } else {
        double err = std::fabs(g - r);
        ok = err <= abs_tol_ + rel_tol_ * std::fabs(r);
        if (err > max_abs_) max_abs_ = err;
      }
      if (!ok) {
        if (bad_here++ == 0)
          fprintf(stderr, "[%s/%s] first mismatch at %zu: gpu=%.9g cpu=%.9g\n",
                  kernel_.c_str(), label, i, g, r);
        n_bad_++;
      }
    }
    n_total_ += gpu.size();
    all_out_.insert(all_out_.end(), gpu.begin(), gpu.end());
  }

  int finish(const char* out_path) {
    FILE* f = fopen(out_path, "wb");
    if (!f) { perror("fopen"); return 2; }
    fwrite(all_out_.data(), sizeof(T), all_out_.size(), f);
    fclose(f);
    printf("{\"kernel\":\"%s\",\"dtype\":\"%s\",\"cases\":%d,\"n\":%zu,"
           "\"n_bad\":%zu,\"max_abs_err\":%.9g,\"abs_tol\":%.9g,"
           "\"rel_tol\":%.9g,\"pass\":%s}\n",
           kernel_.c_str(), dtype_name<T>(), cases_, n_total_, n_bad_,
           max_abs_, abs_tol_, rel_tol_, n_bad_ == 0 ? "true" : "false");
    return n_bad_ == 0 ? 0 : 1;
  }

 private:
  std::string kernel_;
  double abs_tol_, rel_tol_, max_abs_ = 0.0;
  size_t n_bad_ = 0, n_total_ = 0;
  int cases_ = 0;
  std::vector<T> all_out_;
};
