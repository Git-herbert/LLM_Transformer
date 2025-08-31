import torch

data1 = torch.Tensor(1,2,3)
data2 = torch.Tensor(1,2,3)
print(data2.shape)
print(data1.shape)



print(torch.cat([data1, data2], dim=2).shape)