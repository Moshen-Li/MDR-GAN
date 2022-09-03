'''DnCNN模型用来与our‘s model进行对比'''
import argparse
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import os

from model import *
from torch.autograd import Variable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

path = './data/result'
LR_image_dir = './data/train/img'
HR_image_dir = './data/train/gt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH = 1
EPOCHS = 200

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--step', type=int, default=50,
                    help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10')
opt = parser.parse_args()

# 图像处理操作，包括随机裁剪，转换张量
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    y = Image.open(filepath).convert('L')
    return y


class DatasetFromFolder(Dataset):
    def __init__(self, LR_image_dir, HR_image_dir, transforms=None):
        super(DatasetFromFolder, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)])
        self.HR_image_filenames = sorted(
            [os.path.join(HR_image_dir, y) for y in os.listdir(HR_image_dir) if is_image_file(y)])
        self.LR_transform = transform
        self.HR_transform = transform1

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        labels = load_img(self.HR_image_filenames[index])
        HR_images = self.HR_transform(labels)
        LR_images = self.LR_transform(inputs)
        return LR_images, HR_images

    def __len__(self):
        return len(self.LR_image_filenames)


# 构建数据集
processDataset = DatasetFromFolder(LR_image_dir='./data/train/img', HR_image_dir='./data/train/gt', transforms=transform)
trainData = DataLoader(processDataset, batch_size=BATCH)

# 构造模型
netG = DnCNN(1)
netG.to(device)

# 构造迭代器
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# 构造损失函数
lossF = nn.MSELoss().to(device)
cri1 = nn.MSELoss().to(device)
cri2 = nn.L1Loss().to(device)


def adjust_learning_rate(epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    if lr < 1e-6:
        lr = 1e-6
    return lr


for epoch in range(EPOCHS):
    lr = adjust_learning_rate(epoch)  # - 1)
    netG.train()
    processBar = tqdm(enumerate(trainData, 1))

    for param_group in optimizerG.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])

    for i, (LR_images, HR_images) in processBar:
        LR_images = Variable(LR_images.to(device))
        HR_images = Variable(HR_images.to(device))

        fakeImg = netG(LR_images).to(device)

        Loss = cri1(fakeImg, HR_images)

        netG.zero_grad()
        Loss.backward()

        optimizerG.step()

        # 数据可视化
        processBar.set_description(desc='[%d/%d] Loss: %.4f' % (
            epoch, EPOCHS, Loss.item()))

        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(), 'model_save5/netG_epoch_%d_%d.pth' % (4, epoch))
