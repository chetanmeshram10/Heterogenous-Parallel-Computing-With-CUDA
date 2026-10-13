#pragma once
// Kernel from ArraySum.cu. The only change: the element count is a parameter
// instead of the compile-time NUM macro, so it can be tested at many sizes.
__global__ void add(float* A, float* B, float* C, int n)
{
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < (unsigned int)n)
    C[i] = B[i] + A[i];
}
