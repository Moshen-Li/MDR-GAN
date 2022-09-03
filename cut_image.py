'''用来进行图像裁剪'''

import os
import cv2

'''parameter:图片，步长，窗口尺寸（数组），原图高，原图宽，文件名序列，保存路径'''
def sliding_window(image, stepSize, windowSize, height, width, count, save_path):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            if (y + windowSize[1]) <= height and (x + windowSize[0]) <= width:  # 没超出下边界，也超出下边界
                slide = image[y:y + windowSize[1], x:x + windowSize[0], :]
                slide_shrink = cv2.resize(slide, (256, 256), interpolation=cv2.INTER_AREA)
                slide_shrink_gray = cv2.cvtColor(slide_shrink, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(save_path + str(count) + '.png', slide_shrink_gray)
                count = count + 1  # count持续加1
            if (y + windowSize[1]) > height and (x + windowSize[0]) > width:  # 超出右边界，但没超出下边界 或者 超出下边界，但没超出右边界
                continue
            if (y + windowSize[1]) > height and (x + windowSize[0]) <= width:  # 超出下边界，也超出下边界
                break
    return count


# 从文件夹中读取图片(保存图片文件夹的路径,获取第几个图)
def get_image(file_path, index=0):
    # get the image_name list
    img_lst = sorted(list(map(lambda x: os.path.join(file_path, x), os.listdir(file_path))))
    image_path = img_lst[index]
    file_name = image_path.split("\\")[-1]
    # 两种读取方式
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # img = Image.open(image_path)
    return img, file_name, len(img_lst)

def test():
    count = 1
    for i in range(3):
        if i < 8:
            image, name, _ = get_image('./data/clean_original', i)
            root_path = "./data/test/gt/"
            count = sliding_window(image, 256, [256, 256], 640, 400, count, root_path)
        else:
            image, name, _ = get_image('./data/noise_original', i)
            root_path = "./data/test/img/"
            count = sliding_window(image, 256, [256, 256], 640, 400, count, root_path)
