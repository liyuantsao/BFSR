# The code is borrowed from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .models import register

class DenseBlock_5C(nn.Module):
    def __init__(self, nf=3, gc=96, out_dim=96, bias=True):
        super(DenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, out_dim, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= 0.1  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, depth=3, dim=64, bilinear=True):
        super(UNet, self).__init__()
        self.depth = depth
        self.dim = dim
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.input_proj0 = DenseBlock_5C(nf=6, gc=dim, out_dim=dim, bias=True)
        self.input_proj1 = DenseBlock_5C(nf=96, gc=dim, out_dim=dim, bias=True)

        self.down_layers0 = nn.ModuleList()
        for i in range(self.depth):
            if i == self.depth - 1:
                self.down_layers0.append(Down(self.dim * (2 ** i), self.dim * (2 ** (i + 1)) // factor))
            else:
                self.down_layers0.append(Down(self.dim * (2 ** i), self.dim * (2 ** (i + 1))))
                
        self.up_layers0 = nn.ModuleList()
        for i in range(self.depth):
            if i < self.depth - 1:
                self.up_layers0.append(Up(self.dim * (2 ** (self.depth - i)), self.dim * (2 ** (self.depth - i - 1)) // factor, bilinear))
            else:
                self.up_layers0.append(Up(self.dim * (2 ** (self.depth - i)), self.dim * (2 ** (self.depth - i - 1)), bilinear))

        self.down_layers1 = nn.ModuleList()
        for i in range(self.depth):
            if i == self.depth - 1:
                self.down_layers1.append(Down(self.dim * (2 ** i), self.dim * (2 ** (i + 1)) // factor))
            else:
                self.down_layers1.append(Down(self.dim * (2 ** i), self.dim * (2 ** (i + 1))))
                
        self.up_layers1 = nn.ModuleList()
        for i in range(self.depth):
            if i < self.depth - 1:
                self.up_layers1.append(Up(self.dim * (2 ** (self.depth - i)), self.dim * (2 ** (self.depth - i - 1)) // factor, bilinear))
            else:
                self.up_layers1.append(Up(self.dim * (2 ** (self.depth - i)), self.dim * (2 ** (self.depth - i - 1)), bilinear))

        self.inc0 = (DoubleConv(self.dim, self.dim))
        self.inc1 = (DoubleConv(self.dim, self.dim))   

        self.outc0 = (OutConv(self.dim, 6))
        self.outc1 = (OutConv(self.dim, 96))

    def forward(self, epses):
        z0 = self.input_proj0(epses[0])
        z1 = self.input_proj1(epses[1])

        features0 = []
        z0 = self.inc0(z0)
        features0.append(z0)
        for idx, layer in enumerate(self.down_layers0):
            z0 = layer(z0)
            features0.append(z0)

        for idx, layer in enumerate(self.up_layers0):
            z0 = layer(z0, features0[self.depth - 1 - idx])

        features1 = []
        z1 = self.inc1(z1)
        features1.append(z1)
        for idx, layer in enumerate(self.down_layers1):
            z1 = layer(z1)
            features1.append(z1)

        for idx, layer in enumerate(self.up_layers1):
            z1 = layer(z1, features1[self.depth - 1 - idx])

        z0 = self.outc0(z0)
        z1 = self.outc1(z1)

        return [z0, z1]
    
@register('unet')
def make_unet(depth, dim=64, bilinear=True):
    print('UNet: depth={}, dim={}, bilinear={}'.format(depth, dim, bilinear))
    return UNet(depth=depth, dim=dim, bilinear=bilinear)