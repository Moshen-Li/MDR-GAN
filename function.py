"""Dataset.py function"""
# 把多个图像处理操作步骤整合到一起，包括随机裁剪，转换张量
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

# image change
# gray and totensor
from options import args

transform1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图
    # transforms.Resize([500, 952]),
    # transforms.Resize([64,64]),
    # transforms.FiveCrop(64),
    # transforms.RandomCrop(256),
    transforms.ToTensor(),  # 将图像转换为０－２５５范围的张量，进而转化为0.0-1.0的torch张量
])
# gray and totensor
transform2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    # transforms.Resize([500, 952]),
    # transforms.FiveCrop(64),
    # transforms.Resize([64,64]),
    # transforms.RandomCrop(256),
    transforms.ToTensor()
])


# 定义函数判断文件是不是图片，any为或的意思
def is_image_file(filename):
    # endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif"])


# 读取图像，并将其转化为灰度图
def load_img(filepath):
    y = Image.open(filepath).convert('L')
    return y


"""model function"""


# belong to ADNet-conv block
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# belong to ADNet
class blockgroup(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals, stride=1):
        """初始化残差模块"""
        super(blockgroup, self).__init__()
        self.conv = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=stride, padding=2, padding_mode='reflect',
                              bias=False, dilation=2)
        self.bn = nn.BatchNorm2d(outChannals)
        self.relu = nn.LeakyReLU()
        self.aa = blocka(64, 64)
        self.cov11 = conv1x1(192, 64)
        self.cov1 = conv1x1(64, 64, 1)

    def forward(self, x):
        """前向传播过程"""
        res = x
        out1 = self.aa(x)
        # out1 = self.cov1(x)
        out2 = self.aa(out1)
        # out2 = self.cov1(out1)
        out3 = self.aa(out2)
        # out3 = self.cov1(out2)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.cov11(out)
        out = self.relu(out)
        out = out + res
        return out, out1, out2, out3


# belong to ADNet
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


# belong to ADNet
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# belong to blockgroup残差模块
class blocka(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals, stride=1):
        """初始化残差模块"""
        super(blocka, self).__init__()
        self.cov = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dia = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn = nn.BatchNorm2d(outChannals)
        self.relu = nn.LeakyReLU()
        self.cmsm = group_conv(outChannals)
        self.cov11 = conv1x1(128, 64)
        self.sg = nn.Sigmoid()
        self.CBR = ConvBlocka(inChannals, outChannals)
        self.a = ResBlock1(64, 64)
        self.b = ResBlock2(64, 64)

    def forward(self, x):
        """前向传播过程"""
        res = x

        x1 = self.a(x)
        x2 = self.b(x)
        x = torch.cat([x1, x2], 1)
        x = self.cov11(x)
        out = self.relu(x)
        out = out + res
        return out


#  belong to blocka
class group_conv(nn.Module):
    def __init__(self, dim, attention=False):
        super(group_conv, self).__init__()
        self.module1 = Res2NetBottleneck2(dim, dim)
        self.module2 = Res2NetBottleneck2(dim, dim)
        self.module3 = Res2NetBottleneck2(dim, dim)
        self.conv = nn.Conv2d(3 * dim, dim, 1, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.cbam = nn.Sequential(ChannelAttention(3 * dim))
        self.attention = attention

    def forward(self, x):
        y1 = self.module1(x)
        y2 = self.module2(y1)
        y3 = self.module3(y2)
        y = torch.cat((y1, y2, y3), dim=1)
        if self.attention:
            y = self.cbam(y)
        y = self.bn(self.conv(y))

        return self.relu(x + y)


#  belong to blocka
class ConvBlocka(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals, stride=1):
        """初始化残差模块"""
        super(ConvBlocka, self).__init__()
        self.conv = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=stride, padding=1, padding_mode='reflect',
                              bias=False)
        self.bn = nn.BatchNorm2d(outChannals)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """前向传播过程"""
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


#  belong to blocka
class ResBlock1(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock1, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.dia = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """前向传播过程"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        res = x

        x1 = self.conv1(x)
        x1 = self.dia(x1)
        out = self.conv1(x1)
        out = out + res

        out = self.relu(out)
        return out


#  belong to blocka
class ResBlock2(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.dia = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """前向传播过程"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        res = x
        x1 = self.dia(x)
        x1 = self.conv1(x1)
        out = self.dia(x1)

        out = res + out
        out = self.relu(out)
        return out


# belong to group_conv
class Res2NetBottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, se=False, norm_layer=None, shrink=False):
        super(Res2NetBottleneck2, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = planes
        self.conv2 = nn.ModuleList(
            [conv3x3(bottleneck_planes // scales, bottleneck_planes // scales) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
        self.conv3 = conv1x1(bottleneck_planes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales
        self.shrink = nn.Conv2d(inplanes, planes, 1, 1, 0)
        self.bn4 = norm_layer(planes)
        self.skip = shrink

    def forward(self, x):
        identity = x

        xs = torch.chunk(x, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            identity = self.bn4(self.shrink(identity))

        out += identity
        out = self.relu(out)

        return out


# belong to group_conv
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# belong to Res2NetBottleneck2
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


"""loss function"""


class edgeV_loss(nn.Module):
    def __init__(self):
        super(edgeV_loss, self).__init__()

    def forward(self, X1, X2):
        X1_up = X1[:, :, :-1, :]
        X1_down = X1[:, :, 1:, :]
        X2_up = X2[:, :, :-1, :]
        X2_down = X2[:, :, 1:, :]
        return -np.log(int(torch.sum(torch.abs(X1_up - X1_down))) / int(torch.sum(torch.abs(X2_up - X2_down))))


"""train function"""


# 自适应学习率
def adjust_learning_rate(epoch):
    lr = args.lr * (0.1 ** (epoch // args.step))
    if lr < 1e-6:
        lr = 1e-6
    return lr


# computer the loss
def lsgan(real, fake, cri):
    real_label = Variable(torch.ones(real.size()).to("cuda"))
    real_loss = cri(real, real_label)

    fake_label = Variable(torch.zeros(fake.size()).to("cuda"))
    fake_loss = cri(fake, fake_label)

    return 0.5 * (fake_loss + real_loss)


"""test function"""


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img
