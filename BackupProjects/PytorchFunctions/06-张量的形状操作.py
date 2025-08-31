import torch

# data = torch.tensor([[1,2,3],[4,5,6]])
# print(data.shape)
# print(data.size())
# print(data.reshape(3, -1).shape)

# x.squeeze(n) 指定压缩第n位，如果它的维数为1，则压缩，反之不对该维度操作。
# x.unsqueeze(n)表示在第n位的位置添加1维。
data = torch.tensor([1,2,3,4,5])
print(data)
print(data.shape)
print(data.unsqueeze(0))
print(data.unsqueeze(0).shape)
print(data.unsqueeze(0).squeeze())
print(data.unsqueeze(0).squeeze().shape)



# data = torch.Tensor(3,4,5)
# print(data.shape)
# print(torch.transpose(torch.transpose(data, 0, 1),1,2).shape)
# print(torch.permute(data, [1, 2, 0]).shape)
# print(data.permute([1, 2, 0]).shape)

# data = torch.Tensor(2,3)
# print(data.shape)
# print(data.is_contiguous())
# data =data.contiguous()
# print(data.view(3, 2).shape)