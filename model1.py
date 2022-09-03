import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from block import *
import functools
import numpy as np


class ResBlock1(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock1, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """前向传播过程"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResBlock2(nn.Module):
    def __init__(self, input_dim, output_dim, size, strides):
        super(ResBlock2, self).__init__()
        sequence = [nn.Conv2d(input_dim, output_dim, size, stride=strides, padding=int((size - 1) / 2))
            , nn.BatchNorm2d(output_dim)
            , nn.LeakyReLU(inplace=True)]

        self.module = nn.Sequential(*sequence)

    def forward(self, x):
        return self.module(x)


class Generator(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

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

        xa = self.resBlock(x)  # 特征提取单元

        ya = self.SAM(xa)  #

        x1 = self.down(xa)
        x1 = self.CMSM2(x1)

        y1 = self.SAM(x1)

        x2 = self.down1(x1)
        x2 = self.CMSM3(x2)

        y2 = self.SAM(x2)

        x3 = self.up1(x2)
        x4 = x3 + y1

        x4 = self.CMSM2(x4)
        y4 = self.SAM(x4)

        x5 = self.down1(x4)
        x5 = self.CMSM3(x5)
        x6 = x5 + y2

        x6 = self.CMSM3(x6)
        x7 = self.up1(x6)

        x_all = y1 + y4 + x7
        out = self.CMSM4(x_all)
        out = self.up2(out)
        out = out + ya

        out = self.CMSM1(out)
        out = self.cov(out)
        return out + res


class Generator1(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator1, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.cov1 = conv3x3(64, 64, 1)
        self.cov2 = conv3x3(128, 128, 1)
        self.cov3 = conv3x3(256, 256, 1)
        self.cov4 = conv3x3(128, 128, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x  # (4,1,256,256)

        xa = self.resBlock(x)  # (4,64,256,256)特征提取单元

        ya = self.SAM(xa)  # (4,64,256,256)残差

        xa1 = self.down(xa)  # (4,128,128,128)
        # x1 = self.CMSM2(xa1)    # (4,128,128,128)
        x1 = self.cov2(xa1)  # (4,128,128,128)

        y1 = self.SAM(x1)  # (4,128,128,128)

        xa2 = self.down1(x1)  # (4,256,64,64)
        # x2 = self.CMSM3(xa2)    # (4,256,64,64)
        x2 = self.cov3(xa2)  # (4,256,64,64)

        y2 = self.SAM(x2)  # (4,256,64,64)

        x3 = self.up1(x2)  # (4,128,128,128)
        xa4 = x3 + y1  # (4,128,128,128)

        # x4 = self.CMSM2(xa4)    # (4,128,128,128)
        x4 = self.cov2(xa4)  # (4,128,128,128)
        y4 = self.SAM(x4)  # (4,128,128,128)

        xa5 = self.down1(x4)  # (4,256,64,64)

        # x5 = self.CMSM3(xa5)    # (4,256,64,64)
        x5 = self.cov3(xa5)  # (4,256,64,64)
        xa6 = x5 + y2  # (4,256,64,64)

        # x6 = self.CMSM3(xa6)    # (4,256,64,64)
        x6 = self.cov3(xa6)  # (4,256,64,64)
        x7 = self.up1(x6)  # (4,128,128,128)

        x_all = y1 + y4 + x7  # (4,128,128,128)
        # out1 = self.CMSM4(x_all)    # (4,128,128,128)
        out1 = self.cov4(x_all)  # (4,128,128,128)
        out2 = self.up2(out1)  # (4,64,256,256)
        out3 = out2 + ya  # (4,64,256,256)

        # out4 = self.CMSM1(out3)     # (4,64,256,256)
        out4 = self.cov1(out3)  # (4,64,256,256)
        out = self.cov(out4)  # (4,1,256,256)

        return out + res


class Generator2(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator2, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.cov1 = conv3x3(64, 64, 1)
        self.cov2 = conv3x3(128, 128, 1)
        self.cov3 = conv3x3(256, 256, 1)
        self.cov4 = conv3x3(128, 128, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x  # (4,1,256,256)

        xa = self.resBlock(x)  # (4,64,256,256)特征提取单元

        ya = self.SAM(xa)  # (4,64,256,256)残差

        xa1 = self.down(xa)  # (4,128,128,128)
        x1 = self.CMSM2(xa1)    # (4,128,128,128)
        #x1 = self.cov2(xa1)  # (4,128,128,128)

        y1 = self.SAM(x1)  # (4,128,128,128)

        xa2 = self.down1(x1)  # (4,256,64,64)
        x2 = self.CMSM3(xa2)    # (4,256,64,64)
        #x2 = self.cov3(xa2)  # (4,256,64,64)

        y2 = self.SAM(x2)  # (4,256,64,64)

        x3 = self.up1(x2)  # (4,128,128,128)
        xa4 = x3 + y1  # (4,128,128,128)

        # x4 = self.CMSM2(xa4)    # (4,128,128,128)
        x4 = self.cov2(xa4)  # (4,128,128,128)
        y4 = self.SAM(x4)  # (4,128,128,128)

        xa5 = self.down1(x4)  # (4,256,64,64)

        # x5 = self.CMSM3(xa5)    # (4,256,64,64)
        x5 = self.cov3(xa5)  # (4,256,64,64)
        xa6 = x5 + y2  # (4,256,64,64)

        # x6 = self.CMSM3(xa6)    # (4,256,64,64)
        x6 = self.cov3(xa6)  # (4,256,64,64)
        x7 = self.up1(x6)  # (4,128,128,128)

        x_all = y1 + y4 + x7  # (4,128,128,128)
        # out1 = self.CMSM4(x_all)    # (4,128,128,128)
        out1 = self.cov4(x_all)  # (4,128,128,128)
        out2 = self.up2(out1)  # (4,64,256,256)
        out3 = out2 + ya  # (4,64,256,256)

        # out4 = self.CMSM1(out3)     # (4,64,256,256)
        out4 = self.cov1(out3)  # (4,64,256,256)
        out = self.cov(out4)  # (4,1,256,256)

        return out


class Generator3(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator3, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.cov1 = conv3x3(64, 64, 1)
        self.cov2 = conv3x3(128, 128, 1)
        self.cov3 = conv3x3(256, 256, 1)
        self.cov4 = conv3x3(128, 128, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x  # (4,1,256,256)

        xa = self.resBlock(x)  # (4,64,256,256)特征提取单元

        ya = self.SAM(xa)  # (4,64,256,256)残差

        xa1 = self.down(xa)  # (4,128,128,128)
        x1 = self.CMSM2(xa1)    # (4,128,128,128)
        # x1 = self.cov2(xa1)  # (4,128,128,128)

        y1 = self.SAM(x1)  # (4,128,128,128)

        xa2 = self.down1(x1)  # (4,256,64,64)
        x2 = self.CMSM3(xa2)    # (4,256,64,64)
        # x2 = self.cov3(xa2)  # (4,256,64,64)

        y2 = self.SAM(x2)  # (4,256,64,64)

        x3 = self.up1(x2)  # (4,128,128,128)
        xa4 = x3 + y1  # (4,128,128,128)

        x4 = self.CMSM2(xa4)    # (4,128,128,128)
        #x4 = self.cov2(xa4)  # (4,128,128,128)
        y4 = self.SAM(x4)  # (4,128,128,128)

        xa5 = self.down1(x4)  # (4,256,64,64)

        x5 = self.CMSM3(xa5)    # (4,256,64,64)
        # x5 = self.cov3(xa5)  # (4,256,64,64)
        xa6 = x5 + y2  # (4,256,64,64)

        # x6 = self.CMSM3(xa6)    # (4,256,64,64)
        x6 = self.cov3(xa6)  # (4,256,64,64)
        x7 = self.up1(x6)  # (4,128,128,128)

        x_all = y1 + y4 + x7  # (4,128,128,128)
        # out1 = self.CMSM4(x_all)    # (4,128,128,128)
        out1 = self.cov4(x_all)  # (4,128,128,128)
        out2 = self.up2(out1)  # (4,64,256,256)
        out3 = out2 + ya  # (4,64,256,256)

        # out4 = self.CMSM1(out3)     # (4,64,256,256)
        out4 = self.cov1(out3)  # (4,64,256,256)
        out = self.cov(out4)  # (4,1,256,256)

        return out + res


# if __name__ == "__main__":
#    test()


class ConvBlock(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals, stride=1):
        """初始化残差模块"""
        super(ConvBlock, self).__init__()
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu1 = nn.LeakyReLU()

        self.convBlock1 = ConvBlock(64, 64, stride=2)
        self.convBlock2 = ConvBlock(64, 128, stride=1)
        self.convBlock3 = ConvBlock(128, 128, stride=2)
        self.convBlock4 = ConvBlock(128, 256, stride=1)
        self.convBlock5 = ConvBlock(256, 256, stride=2)
        self.convBlock6 = ConvBlock(256, 512, stride=1)
        self.convBlock7 = ConvBlock(512, 512, stride=2)

        self.avePool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(1024, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        x = self.convBlock6(x)
        x = self.convBlock7(x)

        x = self.avePool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


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


def lsgan(real, fake, cri):
    real_label = Variable(torch.ones(real.size())).cuda()
    real_loss = cri(real, real_label)

    fake_label = Variable(torch.zeros(fake.size())).cuda()
    fake_loss = cri(fake, fake_label)

    return 0.5 * (fake_loss + real_loss)


class TV_loss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TV_loss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class edgeV_loss(nn.Module):
    def __init__(self):
        super(edgeV_loss, self).__init__()

    def forward(self, X1, X2):
        X1_up = X1[:, :, :-1, :]
        X1_down = X1[:, :, 1:, :]
        X2_up = X2[:, :, :-1, :]
        X2_down = X2[:, :, 1:, :]
        return -np.log(int(torch.sum(torch.abs(X1_up - X1_down))) / int(torch.sum(torch.abs(X2_up - X2_down))))


def test():
    x = torch.randn((4, 1, 256, 256))
    # out = UnetGenerator(1, 1, 4)
    out = Generator2()
    output = out(x)
    print("ok")
    print(output.shape)


if __name__ == "__main__":
    test()
