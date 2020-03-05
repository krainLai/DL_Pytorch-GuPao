from Get_Sample import Mydataset
from torch.utils import data
import torch
import os
from CNNnetyellow import cnn_net
import torch.nn as nn
import numpy as np
from PIL import ImageDraw,Image

save_path = "models/param_cnn.pt"
if __name__ == '__main__':
    data_set = Mydataset("Mydataset_data/Train_Data")

    net = cnn_net()
    if os.path.isfile(save_path):
        net = torch.load(save_path)
        print("load net...")

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()

    Train = True

    while True:
        if Train:
            train_data = data.DataLoader(dataset=data_set, batch_size=50, shuffle=True)
            for i, (x, y) in enumerate(train_data):
                x = x.permute(0,3,1,2)  # CNN要求输入的数据为：NCHW，data格式为NHWC
                if cuda:
                    x = x.cuda()
                out = net(x)
                if cuda:
                    y = y.cuda()
                loss = loss_func(out, y)

                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 自动求导
                optimizer.step()  # 更新梯度
                if i % 10 == 0:
                    print(loss.item())
                    torch.save(net, save_path)  # 保存模型
        else:
            train_data = data.DataLoader(dataset=data_set, batch_size=1, shuffle=True)
            for i, (x, y) in enumerate(train_data):
                x = x.permute(0, 3, 1, 2)
                if cuda:
                    x = x.cuda()
                out = net(x)

                x = x.permute(0,2,3,1)
                x = x.cpu()
                out_put = out.cpu().detach().numpy()*300
                # out_put = out.detach().numpy() * 300
                y = y.numpy()*300
                img_data = np.array((x[0]*0.5+0.5)*255,dtype=np.int8)
                img = Image.fromarray(img_data,"RGB")
                draw = ImageDraw.Draw(img)
                draw.rectangle(out_put[0],outline="red")#网络输出的结果
                draw.rectangle(y[0], outline="yellow")#原始标签
                img.show()