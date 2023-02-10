#!/usr/bin/env python3
## this sucker produces 120 GFLOPS !! (what does numpy uses for matrix multiplication ??) 
import numpy as np
import os
import time
import torch

N = 4096

A = np.random.randn(N,N).astype(np.float32)
B = np.random.randn(N,N).astype(np.float32)

A1 = torch.randn(N,N, dtype=torch.float32).to("cuda")
B1 = torch.randn(N,N, dtype=torch.float32).to("cuda")

for i in range(1):
  st = time.monotonic()
  res = np.matmul(A,B.T)
  et = time.monotonic()
  gflop = (N*N*2.0*N)*1e-9
  s = et-st

  print(f"{gflop/s= :.4f}")

for i in range(10000):
  st = time.monotonic()
  res = torch.matmul(A1,B1)
  et = time.monotonic()
  gflop = (N*N*2.0*N)*1e-9
  s = et-st

  print(f"torch : {gflop/s= :.4f}")

#with open("./tmp/data","wb") as f:
#  f.write(A.data)
#  f.write(B.data)
#  f.write(res.data)

