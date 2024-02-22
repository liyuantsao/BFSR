# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


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
class ResBlock(nn.Module):
    def __init__(
        self, conv, dim, kernel_size,
        bias=True, bn=False, 
        act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(dim, dim, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(dim))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        self.dim = args.dim
        kernel_size = 3
        scale = args.scale[0]
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, self.dim, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        self.in_chans = args.in_chans
        self.cell_input = args.cell_input # dummy term
        self.cell_dim = 2  # dummy term

        self.input_proj = DenseBlock_5C(nf=self.in_chans, gc=self.dim//2, out_dim=self.dim//2, bias=True)
        self.lr_proj = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_chans, kernel_size=3, stride=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            DenseBlock_5C(nf=self.in_chans, gc=self.dim//2, out_dim=self.dim//2, bias=True)
        )

        # define body module
        self.body = nn.ModuleList()
        for i_layer in range(n_resblocks):
            self.body.append(ResBlock(conv, self.dim, kernel_size, act=act, res_scale=args.res_scale))

        # use last_conv to project back to self.in_chans
        self.last_conv = nn.Conv2d(self.dim, self.in_chans, kernel_size=1)

    def forward(self, x, lr):
        N, C, H, W = x.shape

        x = self.input_proj(x)   # N C H W -> N C=dim H W
        lr_embed = self.lr_proj(lr) # N 3 H W -> N C=dim H W

        if lr_embed.shape != x.shape:
            lr_embed = F.interpolate(lr_embed, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, lr_embed], dim=1)

        # main body of EDSR
        for layer in self.body:
            x = layer(x)

        # final projection
        x = self.last_conv(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('edsr-baseline-latent')
def make_edsr_baseline(in_chans, n_resblocks=16, dim=64, res_scale=1, scale=2, rgb_range=1):
    print(f"in_chans: {in_chans}, n_resblocks: {n_resblocks}, dim: {dim}, res_scale: {res_scale}, scale: {scale}, rgb_range: {rgb_range}")
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.dim = dim
    args.res_scale = res_scale

    args.scale = [scale]

    args.rgb_range = rgb_range
    args.in_chans = in_chans
    args.cell_input = None # dummy term
    args.cell_dim = 2 # dummy term

    return EDSR(args)