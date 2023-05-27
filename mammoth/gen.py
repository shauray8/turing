import numpy as np
import os
import time
from scipy import signal

N = 1024
K = 3

A = np.random.randn(N,N).astype(np.float32)
B = np.random.randn(K,K).astype(np.float32)
C = np.zeros((N-K+1, N-K+1)).astype(np.float32)

ops = (N-K+1)*(N-K+1)*K*K*2.0*K # does not consider padding

def winograd(input, kernel):
    # Get input and kernel dimensions
    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape

    # Determine the output dimensions
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    # Compute the transformed kernel
    transformed_kernel = np.zeros((output_height + kernel_height - 1, output_width + kernel_width - 1))

    for oh in range(output_height):
        for ow in range(output_width):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    transformed_kernel[oh + kh][ow + kw] += input[oh][ow] * kernel[kh][kw]

    # Compute the Winograd convolution result
    output = np.zeros((output_height, output_width))

    for oh in range(output_height):
        for ow in range(output_width):
            for ih in range(0, kernel_height - 1, 2):
                for iw in range(0, kernel_width - 1, 2):
                    output[oh][ow] += (
                        transformed_kernel[oh + ih][ow + iw] +
                        transformed_kernel[oh + ih + 1][ow + iw] +
                        transformed_kernel[oh + ih][ow + iw + 1] +
                        transformed_kernel[oh + ih + 1][ow + iw + 1]
                    )

    print(output)
    return output

st = time.monotonic()
for i in range(N-K+1):
  for j in range(N-K+1):
    temp = 0
    for k in range(K):
      for l in range(K):
        temp += A[i+k][j+l] * B[k][l]

    C[i][j] = temp
et = time.monotonic()
gflop = ops*1e-9
s = et-st
print(f"{gflop/s= :.4f}")
print(C)

"""
for i in range(100):
  st = time.monotonic()
  res = np.matmul(A,B.T)
  et = time.monotonic()
  gflop = (N*N*2.0*N)*1e-9
  s = et-st

  print(f"{gflop/s= :.4f}")
"""
with open("./tmp/data","wb") as f:
  f.write(A.data)
  f.write(B.data)
  f.write(C.data)


