#!/usr/bin/env python3
## this sucker produces 120 GFLOPS !! (what does numpy uses for matrix multiplication ??) 
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time

N = 1024

A = np.random.randn(N,N).astype(np.float32)
B = np.random.randn(N,N).astype(np.float32)

for i in range(1):
  st = time.monotonic()
  res = np.matmul(A,B.T)
  et = time.monotonic()
  gflop = (N*N*2.0*N)*1e-9
  s = et-st

  print(f"{gflop/s= :.4f}")

with open("./tmp/data","wb") as f:
  f.write(A.data)
  f.write(B.data)
  f.write(res.data)

