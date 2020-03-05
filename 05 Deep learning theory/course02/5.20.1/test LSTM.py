import torch.nn as nn
import torch
from torchvision import datasets,transforms

class RNN(nn.Module):
    def __init__(self):
        super.__init__()
        self.rnn = nn.LSTM(input_size=28,hidden_size=64,num_layers=1,batch_first=True)
        self.fc = nn.Linear(64,10)
    def forward(self, x):
        x = x.reshape
lstm = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = lstm(input, (h0, c0))