import torch
import torch.nn  as nn
from torch.utils.data  import DataLoader,Dataset

from torchvision import datasets
from torchvision import transforms
# import os

class LSTM_NET(nn.Module):
    def __init__(self,batch):
        super(LSTM_NET,self).__init__()
        self.batch = batch
        self.lstm = nn.LSTM(28,64,1,batch_first=True)
        self.fc = nn.Linear(64,10)

    def forward(self,x):#x [N,C,H,W]
        x = x.reshape(-1,28,28) #形状变换为 N S V
        h0 = torch.zeros(1,self.batch,64)
        c0 = torch.zeros(1, self.batch,64)

        out,_ = self.lstm(x,(h0,c0)) #out [NSh]

        #此处为重点！
        out = out[:,-1,:]#因为LSTM之需要最后一步,即最后一个序列S的输出, out[N ,H]
        out2 = self.fc(out) # NH->[N,分类]

        active = nn.Softmax(dim=1)
        return active(out2)









if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

    train_dataset = datasets.MNIST(root="./data/",transform=transform,train=True,download=True)
    test_dataset = datasets.MNIST(root="./data/",transform=transform,train=False)

    batch_size,num_works,intput_size = 100,4,28
    train_load = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test_load = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)


    net = LSTM_NET(batch_size)
    loss_funtion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters())

    epochs = 1000
    for epoch in range(epochs):
        for i,(x,y) in enumerate(train_load):
            y_pred = net(x)
            opt.zero_grad()
            loss = loss_funtion(y_pred,y)
            loss.backward()
            opt.step()

            # print(y_pred.shape)
            # print(y.shape)
        for i,(x,y) in enumerate(test_load):
            y_pred = net(x)
            loss = loss_funtion(y_pred,y)
        print(loss)


