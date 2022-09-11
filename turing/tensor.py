import numpy as np
from typing import Optional

class Tensor:
  def __init__(self,data,requires_grad=False,_children=()):
    self.data = data.astype('float32') if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
    self.grad = 0.0
    self.requires_grad = requires_grad
    self._prev = _children
    self._backward = lambda : None

  def __repr__(self):
    return f"Turing_Tensor({self.data}, requires_grad={True if self.requires_grad else False})"

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad = self.requires_grad)
    return Tensor(self.data+other.data)

    def _backward():
      pass

  def __radd__(self, other):
    return self+other

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Tensor(self.data * other.data,requires_grad = (self.requires_grad == True) or (other.requires_grad == True))

  def __rmull__(self, other):
    return self*other

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __truediv__(self, other):
    return self * other ** -1

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Tensor(self.data**other, (self,),f"**{other}")
    return out

  def backward():
    ## some tree node shit gonna copy and paste from karpathy's micrograd
    return 


  def eye(dim, **kwargs): return Tensor(np.eye(dim, dtype=np.float32),**kwargs)

  def zeros(shape, **kwargs): return Tensor(np.zeros(shape, dtype=np.float32),**kwargs)

  def arange(start,stop,step=1,**kwargs): return Tensor(np.arange(start=start, stop=stop, step=step, dtype=np.float32),**kwargs)

  def is_turing_tensor(self): return True if isinstance(self, Tensor) else False

  def ones(size, **kwargs): return Tensor(np.ones(size),**kwargs)
    

if __name__ == "__main__":

  a = Tensor(10,requires_grad=True)
  b = Tensor(99,requires_grad=True)
  c = a + b
  d = c * a**2
  e = d / b
  print(e)
    
