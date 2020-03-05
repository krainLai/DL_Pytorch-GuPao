import torch
import torch.nn as nn

#输入数据
x = torch.randn(2,6,6,6)
conv = nn.Conv2d(6,6,3,1,bias=False)
group_conv = nn.Conv2d(6,9,3,1,groups=3,bias=False)

y = conv(x)
y_group = group_conv(x)
print(x.shape)
print(y.shape)
print(y_group.shape)

params = conv.parameters()
group_params = group_conv.parameters()
print("标准卷积核的参数个数为：{}".format(sum([param.numel() for param in list(params)])))
print("分组卷积核的参数个数为：{}".format(sum([param.numel() for param in list(group_params)])))