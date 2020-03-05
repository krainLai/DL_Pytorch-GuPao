#创建验证码
from PIL import Image,ImageDraw,ImageFont
import random
import os

class drawVerificationCode():
    width,height,picNumber = 0,0,0
    dir = ""

    def __init__(self,width,height,picNumber,dir):
        self.width = width
        self.height = height
        self.picNumber = picNumber
        self.dir = dir

    # 随机字母数字

    def randChar(self,type):
        if type == 0:
            return chr(random.randint(65, 90)) #大写字母
        elif type == 1:
            return chr(random.randint(97, 122)) #小写字母
        else:
            return chr(random.randint(48, 57)) #数字

    # 随机颜色1
    def randColor1(self):
        return (random.randint(64, 255),
                random.randint(64, 255),
                random.randint(64, 255))

    # 随机颜色2
    def randColor2(self):
        return (random.randint(32, 127),
                random.randint(32, 127),
                random.randint(32, 127))

    def draw(self):

        for p in range(self.picNumber):

            # 创建图片对象(画板）
            img = Image.new("RGB", (self.width, self.height), (255, 255, 255))

            # 创建字体对象（字体）
            font = ImageFont.truetype("arial.ttf", 36)

            # 创建Draw对象
            draw = ImageDraw.Draw(img)

            # 填充像素
            for i in range(self.width):
                for j in range(self.height):
                    draw.point((i, j), fill=self.randColor1())

            # 写入文字
            numbers = []
            for i in range(4):
                char = self.randChar(2)
                numbers.append(char)
                draw.text((self.width//4 * i + 10, 10), char, font=font, fill=self.randColor2())
                # draw.text((self.width//4 * i + 10, 10), self.randChar(random.randint(0,2)), font=font, fill=self.randColor2())

           # img.show()
            path = dir  + "{0}.{1}{2}{3}{4}.jpg".format(p+1,numbers[0],numbers[1],numbers[2],numbers[3])
            img.save(path,format="JPEG")

dir = r"Image\\"
ex = os.path.exists(dir)
if ex == False :
    os.mkdir(dir)

dvc = drawVerificationCode(120,60,1000,Image)
dvc.draw()
