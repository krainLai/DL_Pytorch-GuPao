import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self):
        super.__init__()
        self.conver = nn.Conv2d()