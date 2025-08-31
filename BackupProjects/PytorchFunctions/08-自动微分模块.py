import torch
x = torch.tensor(5)
y = torch.tensor(0.)
w = torch.tensor(1.,requires_grad=True,dtype=torch.float32)
b = torch.tensor(3.,requires_grad=True,dtype=torch.float32)
z = x*w +b
loss =torch.nn.MSELoss()
loss =loss(z,y)
loss.backward()
print(w.grad)
print(b.grad)

# x= torch.ones(2,5)
# y = torch.zeros(2,3)
# w = torch.randn(5,3,requires_grad=True) # 设置一个初始w值
# b = torch.randn(3,requires_grad=True) # 设置一个初始b值
# z = torch.matmul(x,w)+b  # 设置计算方式
# loss = torch.nn.MSELoss() # 设置损失函数
# loss =loss(z,y) # 设置损失函数需要对比的值
# loss.backward() # 自动微分
# print(w.grad)
# print(b.grad)