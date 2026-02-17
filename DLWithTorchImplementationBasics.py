import torch
import numpy as np
import pandas as pd
x=torch.tensor([1,2,3])
print("\nValues from the tensor:",x)
print("\nDatatype is:",x.dtype)
numpy_array=np.array([[1,2,3],[4,5,6]])
torch_from_numpy_array=torch.from_numpy(numpy_array)
print("\nTensor from numpy:",torch_from_numpy_array)
zeros=torch.zeros(2,3)
print("\nZeros:",zeros)
ones=torch.ones(2,3)
print("\nOnes:",ones)
randoms=torch.rand(2,3)
print("\nRandoms:",randoms)
aranges=torch.arange(0,10,step=1)
print("\naranges:",aranges)
y=torch.tensor([[1,2,3],[4,5,6]])
print("\nShape of the y:",y.shape)