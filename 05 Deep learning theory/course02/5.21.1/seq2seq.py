#seq2seq网络搭建 编码解码器
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from create_data import MyDataset


img_path ="Image"
save_path = r"seq2seq.pt"

batch_size = 64
num_works=4
epochs = 100

#通过LSTM提取特征
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.lstm = nn.LSTM(128,128,2,batch_first=True)

    def forward(self, x):
        x = x.reshape(-1,180,120).permute(0,2,1) #改变形状，并换轴
        x = x.reshape(-1,180)
        fc1_out = self.fc1(x)
        fc1_out = fc1_out.reshape(-1,120,128)
        out,_ = self.lstm(fc1_out)
        out = out[:,-1,:] #N,128
        return out

#通过LSTM解码
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.lstm = nn.LSTM(128,128,2,batch_first=True)
        self.out = nn.Linear(128,10)
    def forward(self, x):
        x = x.reshape(-1,1,128)
        x = x.expand(batch_size,4,128)# N ,4 ,128
        lstm_out ,_= self.lstm(x)
        out1 = lstm_out.reshape(-1,128)#N*4,128
        out2 = self.out(out1)
        out3 = out2.reshape(-1,4,10) #N,4,10
        return out3

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        encoder = self.encoder(x)
        out = self.decoder(encoder)
        return out


if __name__ == '__main__':

    net = Net()
    if torch.cuda.is_available():
        net = net.cuda()

    opt = torch.optim.Adam([{"params":net.encoder.parameters()},{"params":net.decoder.parameters()}])

    loss_fn = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))

    mydata = MyDataset(img_path)
    # exit()
    train_loader = DataLoader(dataset=mydata,batch_size = batch_size,shuffle=True,drop_last=True,num_workers = num_works)

    for epoch in range(epochs):
        for i ,(x,y) in enumerate(train_loader):
            # print(y.type, " ", x.type)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            out =net(x)
            # print(y.type," ",out.type)
            loss = loss_fn(out,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%5==0:
                test_y = torch.argmax(y,2).detach().cpu().numpy()
                pred_y = torch.argmax(out,2).detach().cpu().numpy()

                acc = np.mean(np.all(pred_y == test_y,axis=1))
                print("epoch:",epoch,"loss:",loss.item(),"acc:",acc)
                print("test_y:",test_y[0],"   pred:",pred_y[0])

        torch.save(net.state_dict(),save_path)



