## this sucker produces 120 GFLOPS !! (what does numpy uses for matrix multiplication ??) 
import numpy as np
import time

N = 1024

A = np.random.randn(N,N).astype(np.float32)
B = np.random.randn(N,N).astype(np.float32)

for i in range(1):
  st = time.monotonic()
  res = np.matmul(A,B)
  et = time.monotonic()
  gflop = (N*N*2.0*N)*1e-9
  s = et-st

  print(f"{gflop/s= :.4f}")

