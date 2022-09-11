import numpy as np
from typing import Optional

class Tensor:
  def __init__(self,data,requires_grad=True,_children=()):
    self.data = data.astype('float32') if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
    self.grad = 0.0
    self.requires_grad = requires_grad
    self._prev = _children
    self._backward = lambda : None

  def __repr__(self):
    return f"Tensor({self.data} with grad {self.grad if self.requires_grad else None})"

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Tensor(self.data+other.data)

    def _backward():
      pass

  def __radd__(self, other):
    return self+other

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Tensor(self.data * other.data)

  def __rmull__(self, other):
    return self*other

  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Tensor(self.data - other.data)

  def backward():
    ## some tree node shit gonna copy and paste from karpathy's micrograd
    return 


  def eye(dim, **kwargs): return Tensor(np.eye(dim, dtype=np.float32),**kwargs)

  def zeros(shape, **kwargs): return Tensor(np.zeros(shape, dtype=np.float32),**kwargs)

  def arange(start,stop,step=1,**kwargs): return Tensor(np.arange(start=start, stop=stop, step=step, dtype=np.float32),**kwargs)

if __name__ == "__main__":
  print(Tensor.arange(-5,5,.2,requires_grad=False))
      
  a = 10+Tensor(10)
  a._backward()
  print(a)

    
