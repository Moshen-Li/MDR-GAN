import os

from torch.utils.data import Dataset

from function import is_image_file, load_img


class Built_Dataset(Dataset):
    def __init__(self, LR_image_dir, HR_image_dir, transform1, transform2):
        super(Built_Dataset, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)])
        self.HR_image_filenames = sorted(
            [os.path.join(HR_image_dir, y) for y in os.listdir(HR_image_dir) if is_image_file(y)])
        # 两种对图像处理的方法
        self.LR_transform = transform1
        self.HR_transform = transform2
    # 用来支持一个整形索引，获取单个数据
    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        labels = load_img(self.HR_image_filenames[index])
        HR_images = self.HR_transform(labels)
        LR_images = self.LR_transform(inputs)
        return LR_images, HR_images

    # 返回DataFrame的大小
    def __len__(self):
        return len(self.LR_image_filenames)
