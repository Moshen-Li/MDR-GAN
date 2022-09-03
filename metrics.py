import argparse
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import skimage.measure as sm
from torchvision.transforms import Compose, ToTensor, Resize
import cv2

parser = argparse.ArgumentParser(description='eval')
parser.add_argument('--falg', type=bool, default=False, help='whether ROI selected')
parser.add_argument('--img_height', type=int, default=300, help='height of HR images')
parser.add_argument('--img_width', type=int, default=300, help='width of HR images')
parser.add_argument('--saved_ROIs_file', type=str, default='./ROIs.npz')
parser.add_argument('--num_sig_ROIs', type=int, default=4)
parser.add_argument('--SR_image_dir', type=str, default='./data/test/gt1')
# parser.add_argument('--LR_image_dir', type=str, default='./data/test/LR')
# parser.add_argument('--GT_image_dir', type=str, default='./data/test/GT')
parser.add_argument('--save_dir', type=str, default='./data/test/gt2')
parser.add_argument('--selected', type=bool, default=False)
opt = parser.parse_args()


# print(len(fig1_sig_region))

def crop(img):
    return transforms.CenterCrop((300, 300))(img)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif"])


def load_img(filepath):
    y = Image.open(filepath).convert('L')
    #y = crop(y)
    return y


SR_image_filenames = sorted(
    [os.path.join(opt.SR_image_dir, x) for x in os.listdir(opt.SR_image_dir) if is_image_file(x)])
# GT_image_filenames = sorted(
#     [os.path.join(opt.GT_image_dir, x) for x in os.listdir(opt.GT_image_dir) if is_image_file(x)])


def draw_roi(img, name, x, y, z, w, img_dir):
    img = img.convert('RGB')
    img = np.asarray(img)
    cv2.rectangle(img, (x, y), (z, w), (0, 255, 0), thickness=2)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    img_path = os.path.join(opt.save_dir, img_dir)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    img = Image.fromarray(img)
    img.save(img_path + '/{}.jpg'.format(name))



def crop_save_ROI(img, name, x, y, z, w, img_dir):
    img = np.asarray(img)
    # if not os.path.exists(roi_save_dir):
    #     os.mkdir(roi_save_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    img_path = os.path.join(opt.save_dir, img_dir)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    bg_img = img[y:w, x:z]
    bg_img = Image.fromarray(bg_img)
    bg_img.save(img_path + '/{}.jpg'.format(name))
    # bg_img_path = os.path.join(roi_save_dir, 'bgROI{}.jpg'.format(name))
    # cv2.imwrite(bg_img_path, bg_img)


def draw():
    for name in sorted(os.listdir(opt.SR_image_dir)):
        SR_img_path = os.path.join(opt.SR_image_dir, name)
        SR_img = load_img(SR_img_path)
        #img_num = len(SR_image_filenames)
        #img_num = int(name.split(".")[0])
        #print(img_num)
        #roi_dir = os.path.join(opt.save_dir, './croped_ROI')
        draw_roi(SR_img, name,40,1,400,'SR')
        crop_save_ROI(SR_img, name,0,122,400,522, 'SR-ROI')


if __name__ == '__main__':
    draw()
