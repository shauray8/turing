#!/usr/bin/env python3
# ToDo - create something more usefull then numpy for turing specifically !
import numpy as np
from typing import Optional

class v_space:
  def __init__(self,data,requires_grad=False,_children=()):
    self.data = data.astype('float32') if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
    self.grad = 0.0
    self.requires_grad = requires_grad
    self._prev = _children
    self._backward = lambda : None

  def __repr__(self):
    return (f"vector space ({self.data}) with grad={True if self.requires_grad else False}\n")

  def __add__(self, other):
    other = other if isinstance(other, v_space) else v_space(other, requires_grad = self.requires_grad)
    return v_space(self.data+other.data)

    def _backward():
      pass

  def __radd__(self, other):
    return self+other

  def __mul__(self, other):
    other = other if isinstance(other, v_space) else v_space(other)
    return v_space(self.data * other.data,requires_grad = (self.requires_grad == True) or (other.requires_grad == True))

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
    out = v_space(self.data**other, (self,),f"**{other}")
    return out

  def backward():
    ## some tree node shit gonna copy and paste from karpathy's micrograd
    return 

  def __matmul__(self,other):
    return mat_mul(self.data,other.data)

  @property
  def shape(self):
    return np.array(self.data).shape

def mat_mul(mat1,mat2):
  return v_space(np.matmul(mat1,mat2))

def eye(dim, **kwargs): return v_space(np.eye(dim, dtype=np.float32),**kwargs)

def zeros(shape, **kwargs): return v_space(np.zeros(shape, dtype=np.float32),**kwargs)

def arange(start,stop,step=1,**kwargs): return v_space(np.arange(start=start, stop=stop, step=step, dtype=np.float32),**kwargs)

def is_turing_tensor(self): return True if isinstance(self, v_space) else False

def ones(*args, **kwargs): return v_space(np.ones(*args),**kwargs)

def randn(*args,**kwargs): return v_space(np.random.randn(*args), **kwargs)

if __name__ == "__main__":
  a = v_space(np.array([1,2,3,4,4,5,6]))
  b = v_space(99)
  c = v_space(99)
  print(a,b,c)
