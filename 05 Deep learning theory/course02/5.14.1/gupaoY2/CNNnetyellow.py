# import torch.nn as nn
# import torch
#
# class cnn_net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1=nn.Sequential(
#             nn.Conv2d(3,16,3,1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#
#         self.layer1=nn.Sequential(
#             nn.Linear(128,64),
#             nn.ReLU(),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(64, 4)
#         )
#     def forward(self, x):
#         cnnout1=self.conv1(x)
#         cnnout2=self.conv2(cnnout1)
#         cnnout3 = self.conv3(cnnout2)
#         cnnout4 = self.conv4(cnnout3)
#         cnnout4=cnnout4.view(cnnout4.size(0),-1)
#         out5=self.layer1(cnnout4)
#         out6=self.layer2(out5)
#
#         # out6_1=out6[:,:4]
#         # out6_2=out6[:,4:]
#         # out6_2=nn.Softmax(out6_2)
#         # out6_2=nn.Sigmoid(out6_2)
#
#         return out6
# if __name__ == '__main__':
#     a = torch.Tensor(2,3,300,300)
#     net = cnn_net()
#     out = net(a)
#     print(out.shape)

import torch.nn as nn
import torch

class cnn_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16,32,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,64,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64,128,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.mlp_layers = nn.Sequential(
            nn.Linear(128*20*20,4)
        )
    def forward(self, x):
        cnn_out = self.cnn_layers(x)
        cnn_out = cnn_out.reshape(-1,128*20*20)
        out = self.mlp_layers(cnn_out)
        return out

if __name__ == '__main__':
    a = torch.Tensor(2,3,300,300)
    net = cnn_net()
    out = net(a)
    print(out.shape)