import torch.nn as nn
import torch
from torchvision import datasets,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
save_path = "param_rnn.pt"

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_size=28,hidden_size=64,num_layers=1,batch_first=True)
        self.fc = nn.Linear(64,10)
    def forward(self, x):
        x = x.reshape(-1,28,28)
        h0 = torch.zeros(1,x.shape[0],64)

        if torch.cuda.is_available():
            h0 = h0.cuda()

        output,_ = self.rnn(x,h0)
        output = self.fc(output[:,-1,:])
        return output

    



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
    data_train = datasets.MNIST(root="./data",transform=transform,train=True,download=True)
    data_test = datasets.MNIST(root="./data",transform=transform,train=False)

    train_load = DataLoader(dataset=data_train,batch_size=64,shuffle=True)
    test_load = DataLoader(dataset=data_test,batch_size=64,shuffle=True)

    model = RNN()
    if os.path.isfile(save_path):
        model = torch.load(save_path)
        print("load net...")

    opt = torch.optim.Adam(model.parameters())
    loss_f = torch.nn.CrossEntropyLoss()

    cuda = torch.cuda.is_available()
    if cuda:
        net = model.cuda()

    epoch_n = 50
    for epoch in range(epoch_n):
        running_loss = 0.0
        running_correct = 0
        testing_correct = 0
        print("-" * 10)
        print("epoch{}/{}".format(epoch,epoch_n))


        for data in train_load:
            X_train,Y_train = data
            X_train,Y_train = Variable(X_train),Variable(Y_train)

            if cuda:
                X_train = X_train.cuda()
                Y_train = Y_train.cuda()

            y_pred = model(X_train)
            loss = loss_f(y_pred,Y_train)
            _,pred = torch.max(y_pred,1)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_correct += torch.sum(pred == Y_train.data)

        print("save model..")
        torch.save(net, save_path)  # 保存模型
        for data in test_load:
            X_test,Y_test = data
            X_test,Y_test = Variable(X_test),Variable(Y_test)
            if cuda:
                X_test = X_test.cuda()
                Y_test = Y_test.cuda()
            outputs = model(X_test)
            _,pred = torch.max(outputs.data,1)
            # print(pred)
            testing_correct +=torch.sum(pred == Y_test.data)

        print("loss:{:.4f}, train acc:{:.4f}, test acc:{}".format(running_loss/len(data_train),100*running_correct/len(data_train),100*testing_correct/len(data_test)))

