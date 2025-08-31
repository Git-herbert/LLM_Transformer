import torch

data = torch.randint(0,10,[4,5,3])
print(data)
print(data[0,:]) # 最外层第0个索引的张量
print("----------------------------")
# print(data[:,0])
# print(data[1,2])

# print(data[[0, 1], [1, 2]])
# print(data[[[0], [1]], [1, 2]])

# print(data[:3, :2])
# print(data[1:, :2])

# print(data[0, :, :])
print(data[:, :, 0])