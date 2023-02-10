import turing
import torch

A = turing.v_space([10])
B = turing.v_space(10)

print(turing.eye(10) @ turing.eye(10))
print(A)
