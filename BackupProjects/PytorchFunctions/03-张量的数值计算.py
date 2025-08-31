import torch


data = torch.randint(0,10,[2,3])
print(data)

# add 10 是乘10？
print(data.add(10)) #数值并不改变
print(data) #数值并不改变

print(data.add_(10)) # 改变原张量
#在 PyTorch 中，add_ 是 torch.Tensor 类的一个方法，用于执行就地（in-place）加法操作。.add_(10) 的调用形式通常是针对一个 Tensor 实例，例如 tensor.add_(10)，其含义是将标量值 10 加到该 Tensor 的每个元素上，并直接修改原 Tensor 的内容，而不创建新的 Tensor。
print(data)
print(data.sub(10))
print(data.mul(10))
print(data.div(10))
print(data.neg()) #加负号


data1 = torch.tensor([[1,2],[3,4]])
data2 = torch.tensor([[5,6],[7,8]])
print(torch.mul(data1, data2)) # 每个对应位置的元素相乘
print(data1 * data2) # 同上


print("---------------------------")
data1 = torch.tensor([[1,2],[3,4],[5,6]])
print(data1)
data2 = torch.tensor([[5,6],[7,8]])
print(data2)
print(data1 @ data2)
print(torch.matmul(data1, data2))
#torch.matmul 是 PyTorch 库中的一个函数，用于计算两个张量的矩阵乘积（matrix product）。它支持多种维度情况下的操作，包括向量、矩阵和批次（batched）张量乘法，并会根据输入张量的形状自动处理广播（broadcasting）和维度调整。