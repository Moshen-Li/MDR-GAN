# -*- coding: utf-8 -*-
# Author：LMS
# noise injection

import cv2
import numpy as np


# canvas = cv2.rectangle(canvas, (50, 50), (100, 100), 255, -1)
# cv2.imshow("Square", canvas)
# cv2.waitKey()
# cv2.destoryALLWindows()

def draw(num, file_path, width, height):
    # 绘制形状:矩形，圆形，圆弧
    flag = 0  # 形状选择器
    for i in range(1, num + 1):
        # 生成画布
        canvas = np.zeros((height, width, 1), np.uint8)
        if flag == 0:
            # 绘制矩形
            # 随机生成边长边长70-150
            le_height = np.random.randint(70, 150)
            le_weight = np.random.randint(70, 150)
            # 随机生成点
            point1_1 = np.random.randint(0, 256 - le_weight)
            point1_2 = np.random.randint(0, 256 - le_height)
            point2_1 = le_weight + point1_1
            point2_2 = le_height + point1_2
            clean_image = draw_rectangle(canvas, point1_1, point1_2, point2_1, point2_2)
            flag += 1
        elif flag == 1:
            # 绘制圆形
            # 随机生成半径
            radius = np.random.randint(60, 80)
            # 随机生成圆心
            point_center1 = np.random.randint(radius, 256 - radius)
            point_center2 = point_center1
            clean_image = draw_circle(canvas, point_center1, point_center2, radius)
            flag += 1
        elif flag == 2:
            # 绘制圆弧
            # 随机生成半径
            length1 = np.random.randint(60, 80)
            length2 = np.random.randint(60, 80)
            radius = max(length1, length2)
            # 随机生成圆心
            point_center1 = np.random.randint(radius, 256 - radius)
            point_center2 = point_center1
            # 随机生成偏转角度、起始角、终止角
            angle1 = np.random.randint(0, 180)
            angle2 = np.random.randint(0, 60)
            angle3 = np.random.randint(120, 180)
            # 1目标图片；2圆心；3轴的长度(长轴，短轴)；4偏转角度；5圆弧的起始角度；6终止角度；7颜色；8内容是否填充。
            clean_image = draw_ellipse(canvas, point_center1, point_center2, length1, length2, angle1, angle2,
                                      angle3)
            flag = 0
        # 添加噪声
        noise_img = gauss_Noise(clean_image, 0.2, 0.3)
        # 保存图像
        filename_noise = file_path + 'noise/' + str(i).zfill(3) + '.jpg'
        cv2.imwrite(filename_noise, noise_img)
        filename_clean = file_path + 'clean/' + str(i).zfill(3) + '.jpg'
        cv2.imwrite(filename_clean, clean_image)


# 绘制矩形
def draw_rectangle(canvas, point1_1, point1_2, point2_1, point2_2):
    canvas = cv2.rectangle(canvas, (point1_1, point1_2), (point2_1, point2_2), 255, -1)
    return canvas


# 绘制圆形
def draw_circle(image, point_center1, point_center2, radius):
    canvas = cv2.circle(image, (point_center1, point_center2), radius, 255, -1)
    return canvas


# 绘制圆弧
def draw_ellipse(image, point_center1, point_center2, length1, length2, angle1, angle2, angle3):
    # 1目标图片；2圆心；3轴的长度(长轴，短轴)；4偏转角度；5圆弧的起始角度；6终止角度；7颜色；8内容是否填充。
    canvas = cv2.ellipse(image, (point_center1, point_center2), (length1, length2), angle1, angle2, angle3, 255, -1)
    return canvas


# 添加高斯噪声
def gauss_Noise(image, mu, sigma):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, 0, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise


# 添加椒盐噪声
# def sp_Noise(img,snr):
#     h = img.shape[0]
#     w = img.shape[1]
#     img1 = img.copy()
#     sp = h * w  # 计算图像像素点个数
#     NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
#     for i in range(NP):
#         randx = np.random.randint(1, h - 1)  # 生成一个 1 至 h-1 之间的随机整数
#         randy = np.random.randint(1, w - 1)  # 生成一个 1 至 w-1 之间的随机整数
#         if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
#             img1[randx, randy] = 0
#         else:
#             img1[randx, randy] = 255
#     return img1


if __name__ == "__main__":
    # 图像保存路径
    file_path = 'data/noise injection/test/'
    # 绘制图像数量
    num = 10
    # 图像大小
    width = 256
    height = 256
    draw(num, file_path, width, height)
