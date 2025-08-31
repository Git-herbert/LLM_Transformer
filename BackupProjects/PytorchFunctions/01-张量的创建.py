import torch
import numpy as np

print(torch.tensor(10))

print(np.random.randn(2, 3))
print(type(np.random.randn(2, 3))) #  class 'numpy.ndarray' 两个方括号
print(type(np.random.randn(2, 3)[1]))

print(torch.tensor(np.random.randn(2, 3)))
#在 NumPy 中，np.random 是一个包（package），其 __init__.py 文件负责初始化并暴露随机数生成相关的功能。具体来说，np.random.randn 函数是通过 from .mtrand import * 语句从子模块 mtrand 中导入的。这使得 mtrand 中的所有函数（包括 randn）直接在 np.random 命名空间下可用，而无需显式引用 mtrand。

print(torch.tensor([[2.5, 3, 4], [3, 4, 5]]))

print(torch.Tensor(3, 3))

print(torch.Tensor([10]))

print(torch.Tensor(np.random.randn(2, 3)))

print(torch.Tensor([[2.5, 3, 4], [3, 4, 5]]))

print(torch.arange(0, 10, 2))
print(torch.linspace(0, 10, 11))
print(torch.randn(2, 3))


print(torch.zeros(2, 3))
print(torch.ones(2, 3))
print(torch.full((2, 3), 10))

data = torch.full((2,3),10)
print(data.dtype)
data1 =data.type(torch.DoubleTensor)
# print(data1.dtype)
# data2 =data.double()
# print(data2.dtype)
