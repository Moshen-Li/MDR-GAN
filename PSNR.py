import cv2
import numpy as np
import math

import torch
from torch.autograd import Variable

# im1 = Image.open("./data/test/gt/AVG_lms7.bmp")
# # im1 = im1.convert('L')
# im2 = Image.open("./data/test/denoise/007_resultSR2.bmp")
# image_path1 = "D:\\chenkun\\ADnet\\root\\data\\GT\\1.jpg"
# image_path2 = "D:\\chenkun\\ADnet\\root\\data\\LR\\1_resultSR1.jpg"

# 因为是张彩色图片所以截取出一个通道
# gt = cv2.imread(image_path1)
# img= cv2.imread(image_path2)
gt = cv2.imread("./data/test/gt/AVG_lms7.bmp")
img= cv2.imread("./data/test/denoise/007_resultSR1.bmp")

def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

print(psnr2(gt,img))



