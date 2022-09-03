import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from model import *

'''test'''
def imshow(path, i, save_path, net):
    """展示结果"""
    net.eval()
    preTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        #transforms.Resize([400, 640]),
        transforms.Resize([256,256]),
        # transforms.RandomCrop(64),
        transforms.ToTensor()
    ])

    pilImg = Image.open(path)
    img = preTransform(pilImg).unsqueeze(0)

    source = net(img)[0, :, :, :]
    source = source.cpu().detach().numpy()  # 转为numpy
    source = source.transpose((1, 2, 0))  # 切换形状
    source = np.clip(source, 0, 1)  # 修正图片
    source = np.squeeze(source)
    plt.imshow(source)
    img = Image.fromarray(np.uint8(source * 255))
    img = img.resize((pilImg.width, pilImg.height), Image.BICUBIC)
    img = img.convert('L')
    img.save(save_path  +str(i).zfill(3) + ".jpg")  # 将数组保存为图片


if __name__ == '__main__':
    for i in range(3):
        num = 7 + i
        imshow("./data/test/img/00" + str(num) + ".bmp", net)
