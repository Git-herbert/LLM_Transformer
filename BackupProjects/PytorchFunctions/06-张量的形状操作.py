import torch

# data = torch.tensor([[1,2,3],[4,5,6]])
# print(data.shape)
# print(data.size())
# print(data.reshape(3, -1).shape)

# data = torch.tensor([1,2,3,4,5])
# print(data.shape)
# print(data.unsqueeze(0).shape)
# print(data.unsqueeze(0).squeeze().shape)

# data = torch.Tensor(3,4,5)
# print(data.shape)
# print(torch.transpose(torch.transpose(data, 0, 1),1,2).shape)
# print(torch.permute(data, [1, 2, 0]).shape)
# print(data.permute([1, 2, 0]).shape)

data = torch.Tensor(2,3)
print(data.shape)
print(data.is_contiguous())
data =data.contiguous()
print(data.view(3, 2).shape)