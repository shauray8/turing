import numpy as np
import os
import time
from scipy import signal

N = 4
K = 3

A = np.random.randn(N,N).astype(np.float32)
B = np.random.randn(K,K).astype(np.float32)
C = np.zeros((N-K+1,N-K+1)).astype(np.float32)

ops = (N-K+1)*(N-K+1)*K*K*2.0*K # does not consider padding

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


