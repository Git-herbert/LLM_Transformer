import torch
import numpy as np

data =torch.tensor([2,3,4])
data_numpy =data.numpy()
print(type(data))
print(type(data_numpy))

data = np.array([2,3,4])
print(type(data))
print(type(torch.from_numpy(data)))
print(type(torch.tensor(data)))

data = torch.tensor([30,])
print(data.item())