import os
import  torch
import numpy as np
from PIL import  Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

#data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

class MyDataset(Dataset):
    def __init__(self,root):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        self.list=[]
        for filename in os.listdir(root):
            x = os.path.join(root,filename)
            ys = filename.split(".")

            # print(ys)
            y = self.one_hot(ys[1]) #获取标签
            self.list.append([x,np.array(y)])

    def __len__(self):
        return len(self.list)

    def __getitem__(self,index):
        img_path,label = self.list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        label=torch.from_numpy(label)
        return img,label

    def one_hot(self,x):
        #z = np.zeros(shape=[4,10],type =np.float)
        z = torch.zeros(4,10)
        for i in range(4):
            #print(x[i])
            index = int(x[i])
            z[i][index]+=1
        return z



if __name__ == "__main__":

    mydata = MyDataset("Image")
    data_loader = DataLoader(mydata,batch_size=16,shuffle=True)

    for i ,(x,y) in enumerate(data_loader):
        print(i," ",x,y)
        break

