# 以tiff转png为例，其他格式同理，
# 代码中路径更改为自己图像存放路径即可
from PIL import Image
import os

imagesDirectory= r"C:\Users\YangYang\Desktop\picture\1\tif"  # tiff图片所在文件夹路径
distDirectory = os.path.dirname(imagesDirectory)
distDirectory = os.path.join(distDirectory, "png")# 要存放png格式的文件夹路径
for imageName in os.listdir(imagesDirectory):
    imagePath = os.path.join(imagesDirectory, imageName)
    image = Image.open(imagePath)# 打开tiff图像
    distImagePath = os.path.join(distDirectory, imageName[:-4]+'.bmp')# 更改图像后缀为.png，与原图像同名
    image.save(distImagePath)# 保存png图像