import torch

data = torch.randint(0,10,[2,3],dtype=torch.float64)
print(data)
print(data.mean()) #对每个元素加和除于元素个数
print(data.sum()) # 对各个元素加和
print(torch.pow(data, 2)) # 用于对输入张量 data 的每个元素进行幂运算，这里是将每个元素平方（即计算每个元素的 2 次方）。
print(data.sqrt()) # 每个元素计算其平方根
print(data.exp()) # 以e为底，数字为幂
print(data.log()) # 以loge(data)
print(data.log2())
print(data.log10())