import os
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageDraw
import numpy as np

class Mydataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.dataset=os.listdir(self.path)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        lable=torch.Tensor(np.array(self.dataset[index].split(".")[1:5],dtype=np.float32)/300)

        img_path=os.path.join(self.path,self.dataset[index])
        img=Image.open(img_path)
        img_data=torch.Tensor((np.array(img)/255-0.5)/0.5)

        return img_data,lable

if __name__=="__main__":
    mydata=Mydataset(r"Mydataset_data\Train_Data")

    dataloader=DataLoader(dataset=mydata,batch_size=1,shuffle=True)

    for i,(x,y) in enumerate(dataloader):
      print(x.size())
      print(y.size())
      x=x[0].numpy()
      y=y[0].numpy()
      #print(y)
      img_data=np.array((x*0.5+0.5)*255,dtype=np.uint8)
      img=Image.fromarray(img_data,"RGB")
      draw=ImageDraw.Draw(img)
      draw.rectangle(y*300,outline="red")
      img.show()