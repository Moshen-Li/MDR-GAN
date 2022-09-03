
"discriminator model"
import functools
import time

import torch
from torch import nn

from function import conv1x1, blockgroup, SpatialAttention, ChannelAttention, ConvBlocka, conv3x3


# class PatchGAN(nn.Module):
#     """Defines a PatchGAN discriminator. """
#
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator.
#
#         When n_layers = 3, this ia 70x70 PatchGAN
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(PatchGAN, self).__init__()  # 调用父类的构造函数
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         # output 1 channel prediction map
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)
class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator. """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator.

        When n_layers = 3, this ia 70x70 PatchGAN

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGAN, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

    # 网络整体结构如下
    """[Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), LeakyReLU(negative_slope=0.2, inplace=True), 
    Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
    Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
    Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)), InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
    Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))]"""

"Generate model"
class ADNet(nn.Module):
    """生成模型(4x)"""
    def __init__(self):
        """初始化模型配置"""
        super(ADNet, self).__init__()
        # 卷积模块1
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.cov = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.cov11 = conv1x1(64, 1)
        self.block = blockgroup(64, 64)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.bn = nn.BatchNorm2d(64)
        self.conv1 = ConvBlocka(1, 64)


    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x

        x = self.conv1(x)
        x, out1, out2, out3 = self.block(x)
        x = self.cov11(x)

        out = x + res
        return out

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return x + out

def test():
    begin = time.time()
    x = torch.randn((2, 1, 256, 256))
    net = ADNet()
    out, out1, out2, out3 = net(x)
    print(time.time() - begin)
    print(out.shape)


if __name__ == "__main__":
    test()