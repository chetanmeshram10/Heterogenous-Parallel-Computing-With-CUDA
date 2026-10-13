#pragma once
// Kernel moved verbatim from template.cu, plus ONE added line: the bounds guard.
// Compile with -DBLUR_NO_GUARD to restore the original (unguarded) behaviour;
// see verification/FINDINGS.md for why that is a bug.

#define BLUR_SIZE 1
#define isValid(X, Y) (X >= 0 && Y>=0 && X < height && Y < width)

__global__ void blur(float* input, float* output, int height, int width)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

#ifndef BLUR_NO_GUARD
    if (X >= height || Y >= width) return;   // added: threads outside the image must not write
#endif

    float ans1=0, ans2 = 0, ans3 = 0;
    int count=0;
    for(int i =  X - 1; i <= X+1; i++)
      for(int j = Y - 1; j<= Y+1; j++)
      {
        if(isValid(i, j))
          { ans1 += input[3 * (i*width + j)];
            ans2 += input[3 * (i*width + j) + 1];
            ans3 += input[3 * (i*width + j) + 2];
            count++;
          }
      }

    ans1 = ans1/ count;
    ans2 = ans2/ count;
    ans3 = ans3/ count;

    output[ 3 * (X*width + Y)] = ans1;
    output[ 3 * (X*width + Y) + 1] = ans2;
    output[ 3 * (X*width + Y) + 2] = ans3;
}
