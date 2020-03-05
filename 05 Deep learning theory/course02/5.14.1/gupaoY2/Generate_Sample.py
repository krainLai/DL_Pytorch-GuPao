import os
from PIL import Image
import numpy as np

bg_path=r"Mydataset_data\bg_pic"
yellow_path=r"Mydataset_data\yellow"
train_path=r"Mydataset_data\Train_Data"
validate_path=r"Mydataset_data\Validate_Data"
test_path=r"Mydataset_data\Test_Data"

x=1
for filename in os.listdir(bg_path):
    print(filename)
    background=Image.open(r"{0}/{1}".format(bg_path,filename))
    #background.show()

    background=background.convert("RGB")

    # 图像统一大小
    bgresize=background.resize((300,300))
    # print(bgresize)

    #bgresize.save(r"{0}/{1}.png".format(train_path, str(x) + "." + str(0) +
                                        # "." + str(0) + "." + str(0) + "." + str(0) + "." + "0"))
    # 随机粘贴
    name=np.random.randint(1,21)
    yellow_img=Image.open(r"{0}/{1}.png".format(yellow_path,name))
    #print(yellow_img)

    ran_w = np.random.randint(50,180)
    ran_h = np.random.randint(50, 180)
    yellow_img_new1=yellow_img.resize((ran_w,ran_h)) #小黄人图像随机压缩

    rota_value=np.random.randint(-45,45)
    yellow_img_new2=yellow_img_new1.rotate(rota_value) #随机旋转图像
    #yellow_img_new2.show()

    #随机粘贴坐标
    ran_x1=np.random.randint(0,300-ran_w)
    ran_y1 = np.random.randint(0, 300 - ran_h)

    r,g,b,a=yellow_img_new2.split()

    bgresize.paste(yellow_img_new2,(ran_x1,ran_y1),mask=a)
   # bgresize.show()

    ran_x2=ran_x1+ran_w
    ran_y2 = ran_y1 + ran_h

    bgresize.save(r"{0}/{1}.png".format(train_path,str(x)+"."+str(ran_x1)+
                                        "."+str(ran_y1)+"."+str(ran_x2)+"."+str(ran_y2)+"."+"1"))
    # bgresize.save(r"{0}/{1}.png".format(validate_path, str(x) + "." + str(ran_x1) +
    #                                     "." + str(ran_y1) + "." + str(ran_x2) + "." + str(ran_y2) + "." + "1"))
    # bgresize.save(r"{0}/{1}.png".format(test_path, str(x) + "." + str(ran_x1) +
    #                                     "." + str(ran_y1) + "." + str(ran_x2) + "." + str(ran_y2) + "." + "1"))
    x+=1

    if x==500:
        break