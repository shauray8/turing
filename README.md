# Turing
Turing is an Autograd Engine which as said computes gradients for your neural network.

## Example
```
from turing import Tensor

x = Tensor(10) + Tensor(20)
y = Tensor(10) * 99
z = Tensor.eye(2, requires_grad=False)

## Tensor is just a numpy array with gradients
```
** updating readme 
