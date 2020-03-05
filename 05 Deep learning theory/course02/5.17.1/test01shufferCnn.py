#深度神经网络设计
import torch
import torchvision


import torchvision.datasets as dset



dataset = dset.ImageFolder('./data/') #没有transform，先看看取得的原始图像数据
print(dataset.classes)  #根据分的文件夹的名字来确定的类别
print(dataset.class_to_idx) #按顺序为这些类别定义索引为0,1...
print(dataset.imgs) #返回从所有文件夹中得到的图片的路径以及其类别
#通道分组之后，有通道不融合的缺陷，所以引入通道清洗的概念

#pytorch中的通道清洗
torchvision.models.shufflenet_v2_x2_0()