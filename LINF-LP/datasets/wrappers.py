import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F

from datasets import register
from utils import to_pixel_samples
from utils import make_coord

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        lr_up = F.interpolate(((crop_lr - 0.5) / 0.5).unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_down = F.interpolate(lr_up.unsqueeze(0), crop_lr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_residual = lr_up - F.interpolate(lr_up_down.unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
            lr_up_residual = lr_up_residual[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_lr_up': lr_up_residual
        }

@register('sr-implicit-paired-fast-patch')
class SRImplicitPairedFastPatch(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, patch_size=3):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        lr_up = F.interpolate(((crop_lr - 0.5) / 0.5).unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_down = F.interpolate(lr_up.unsqueeze(0), crop_lr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_residual = lr_up - F.interpolate(lr_up_down.unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]

        h, w, _ = hr_coord.shape
        pad_h = self.patch_size-h%self.patch_size
        pad_w = self.patch_size-w%self.patch_size
        coord_pad = F.pad(hr_coord.permute(2, 0, 1), (0, pad_w, 0, pad_h), "constant", 0)
        coord_unfold = coord_pad.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        coord_patch_center = coord_unfold[:, :, :, self.patch_size//2, self.patch_size//2].permute(1, 2, 0)

        ps = self.patch_size
        lr_up_patch = F.pad(lr_up_residual, (0, pad_w, 0, pad_h), "constant", 0).unfold(1, ps, ps).unfold(2, ps, ps)
        c, w, h, a, b = lr_up_patch.shape
        ps_square = ps*ps
        lr_up_patch = lr_up_patch.contiguous().view(c, w, h, ps_square).permute(0, 3, 1, 2).contiguous().view(c*ps_square, w, h)

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'gt_lr_up': lr_up_patch,
            'coord': coord_patch_center,
            'cell': cell,
            'gt': hr_rgb
        }
    
    
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        lr_up = F.interpolate(((crop_lr - 0.5) / 0.5).unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_down = F.interpolate(lr_up.unsqueeze(0), crop_lr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_residual = lr_up - F.interpolate(lr_up_down.unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        hr_rgb_residual = ((crop_hr - 0.5) / 0.5) - lr_up
        
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)

            lr_up_residual = lr_up_residual.contiguous().view(lr_up_residual.shape[0], -1)
            lr_up_residual = lr_up_residual[:, idx]
            lr_up_residual = lr_up_residual.view(lr_up_residual.shape[0], h_lr, w_lr)

            hr_rgb_residual = hr_rgb_residual.contiguous().view(hr_rgb_residual.shape[0], -1)
            hr_rgb_residual = hr_rgb_residual[:, idx]
            hr_rgb_residual = hr_rgb_residual.view(hr_rgb_residual.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_pixel': hr_rgb_residual,
            'gt_lr_up': lr_up_residual
        }    


@register('sr-implicit-downsampled-fast-patch')
class SRImplicitDownsampledFastPatch(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, patch_size=3):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        lr_up = F.interpolate(((crop_lr - 0.5) / 0.5).unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        hr_rgb_residual = ((crop_hr - 0.5) / 0.5) - lr_up
        hr_rgb = crop_hr

        hr_rgb_patch = F.pad(hr_rgb_residual, (self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2), "constant", 0).unfold(1, self.patch_size, 1).unfold(2, self.patch_size, 1)
        c, w, h, a, b = hr_rgb_patch.shape
        ps_square = self.patch_size*self.patch_size
        hr_rgb_patch = hr_rgb_patch.contiguous().view(c, w, h, ps_square).permute(0, 3, 1, 2).contiguous().view(c*ps_square, w, h)
        
        lr_up_down = F.interpolate(lr_up.unsqueeze(0), crop_lr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_residual = lr_up - F.interpolate(lr_up_down.unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

        lr_up_patch = F.pad(lr_up_residual, (self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2), "constant", 0).unfold(1, self.patch_size, 1).unfold(2, self.patch_size, 1)
        c, w, h, a, b = lr_up_patch.shape
        lr_up_patch = lr_up_patch.contiguous().view(c, w, h, ps_square).permute(0, 3, 1, 2).contiguous().view(c*ps_square, w, h)

        if self.inp_size is not None:
            # TODO: do not sample border pixels...
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb_patch = hr_rgb_patch.contiguous().view(hr_rgb_patch.shape[0], -1)
            hr_rgb_patch = hr_rgb_patch[:, idx]
            hr_rgb_patch = hr_rgb_patch.view(hr_rgb_patch.shape[0], h_lr, w_lr)

            lr_up_patch = lr_up_patch.contiguous().view(lr_up_patch.shape[0], -1)
            lr_up_patch = lr_up_patch[:, idx]
            lr_up_patch = lr_up_patch.view(lr_up_patch.shape[0], h_lr, w_lr)

            hr_rgb = hr_rgb.contiguous().view(hr_rgb.shape[0], -1)	
            hr_rgb = hr_rgb[:, idx]	
            hr_rgb = hr_rgb.view(hr_rgb.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,   # for eval during training: only eval central pixels
            'gt_patch': hr_rgb_patch,    # for training: [-1, 1], residual
            'gt_lr_up': lr_up_patch  # for training: [-1, 1], residual
        }    


@register('sr-implicit-downsampled-fast-patch-test')
class SRImplicitDownsampledFastPatchTest(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, patch_size=3):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        lr_up = F.interpolate(((crop_lr - 0.5) / 0.5).unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_down = F.interpolate(lr_up.unsqueeze(0), crop_lr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_residual = lr_up - F.interpolate(lr_up_down.unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]

        h, w, _ = hr_coord.shape
        if h%self.patch_size:
            pad_h = self.patch_size-h%self.patch_size
        else:
            pad_h = 0
        if w%self.patch_size:
            pad_w = self.patch_size-w%self.patch_size
        else:
            pad_w = 0
        coord_pad = F.pad(hr_coord.permute(2, 0, 1), (0, pad_w, 0, pad_h), "constant", 0)
        coord_unfold = coord_pad.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        coord_patch_center = coord_unfold[:, :, :, self.patch_size//2, self.patch_size//2].permute(1, 2, 0)

        ps = self.patch_size
        lr_up_patch = F.pad(lr_up_residual, (0, pad_w, 0, pad_h), "constant", 0).unfold(1, ps, ps).unfold(2, ps, ps)
        c, w, h, a, b = lr_up_patch.shape
        ps_square = ps*ps
        lr_up_patch = lr_up_patch.contiguous().view(c, w, h, ps_square).permute(0, 3, 1, 2).contiguous().view(c*ps_square, w, h)

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'gt_lr_up': lr_up_patch,
            'coord': coord_patch_center,
            'cell': cell,
            'gt': hr_rgb
        }
        

@register('sr-implicit-downsampled-fast-crop')
class SRImplicitDownsampledFastCrop(Dataset):

    def __init__(self, dataset, inp_size=48, scale_max=4, augment=False):
        self.dataset = dataset
        self.out_size = inp_size
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(1, self.scale_max)
        
        h_lr = self.out_size
        w_lr = self.out_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - h_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = resize_fn(crop_hr, (h_lr, w_lr))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        lr_up = F.interpolate(((crop_lr - 0.5) / 0.5).unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_down = F.interpolate(lr_up.unsqueeze(0), crop_lr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_residual = lr_up - F.interpolate(lr_up_down.unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        hr_rgb_residual = ((crop_hr - 0.5) / 0.5) - lr_up

        x0 = random.randint(0, hr_rgb.shape[-2] - self.out_size)
        y0 = random.randint(0, hr_rgb.shape[-1] - self.out_size)
        hr_rgb = hr_rgb[:, x0: x0 + self.out_size, y0: y0 + self.out_size]
        hr_coord = hr_coord[x0: x0 + self.out_size, y0: y0 + self.out_size, :]
        lr_up_residual = lr_up_residual[:, x0: x0 + self.out_size, y0: y0 + self.out_size]
        hr_rgb_residual = hr_rgb_residual[:, x0: x0 + self.out_size, y0: y0 + self.out_size]

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_lr_up': lr_up_residual, 
            'gt_pixel': hr_rgb_residual 
        }    



@register('sr-implicit-downsampled-fast-crop-patch')
class SRImplicitDownsampledFastCropPatch(Dataset):
    def __init__(self, dataset, inp_size=48, scale_max=4, augment=False, patch_size=3):
        self.dataset = dataset
        # self.out_size = inp_size
        self.out_size = inp_size * patch_size
        self.scale_max = scale_max
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(1, self.scale_max)

        h_lr = self.out_size
        w_lr = self.out_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - h_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = resize_fn(crop_hr, (h_lr, w_lr))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # generate the training data of HR image
        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        lr_up = F.interpolate(((crop_lr - 0.5) / 0.5).unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        hr_rgb_residual = ((crop_hr - 0.5) / 0.5) - lr_up
        
        hr_rgb_patch = F.pad(hr_rgb_residual, (self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2), "constant", 0).unfold(1, self.patch_size, 1).unfold(2, self.patch_size, 1)
        c, w, h, a, b = hr_rgb_patch.shape
        ps_square = self.patch_size*self.patch_size
        hr_rgb_patch = hr_rgb_patch.contiguous().view(c, w, h, ps_square).permute(0, 3, 1, 2).contiguous().view(c*ps_square, w, h)

        x0 = random.randint(0, hr_rgb.shape[-2] - self.out_size)
        y0 = random.randint(0, hr_rgb.shape[-1] - self.out_size)
        hr_coord = hr_coord[x0: x0 + self.out_size, y0: y0 + self.out_size, :]
        hr_rgb = hr_rgb[:, x0: x0 + self.out_size, y0: y0 + self.out_size]

        hr_rgb_patch = hr_rgb_patch[:, x0: x0 + self.out_size, y0: y0 + self.out_size]
        hr_rgb_patch_unfold = hr_rgb_patch.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        hr_rgb_patch_center = hr_rgb_patch_unfold[:, :, :, self.patch_size//2, self.patch_size//2]

        # generate the same training data of upsampled LR
        lr_up_down = F.interpolate(lr_up.unsqueeze(0), crop_lr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        lr_up_residual = lr_up - F.interpolate(lr_up_down.unsqueeze(0), crop_hr.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

        lr_up_patch = F.pad(lr_up_residual, (self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2), "constant", 0).unfold(1, self.patch_size, 1).unfold(2, self.patch_size, 1)
        c, w, h, a, b = lr_up_patch.shape
        lr_up_patch = lr_up_patch.contiguous().view(c, w, h, ps_square).permute(0, 3, 1, 2).contiguous().view(c*ps_square, w, h)

        lr_up_patch = lr_up_patch[:, x0: x0 + self.out_size, y0: y0 + self.out_size]
        lr_up_patch_unfold = lr_up_patch.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        lr_up_patch_center = lr_up_patch_unfold[:, :, :, self.patch_size//2, self.patch_size//2]
        
        h, w, _ = hr_coord.shape
        if h%self.patch_size:
            pad_h = self.patch_size-h%self.patch_size
        else:
            pad_h = 0
        if w%self.patch_size:
            pad_w = self.patch_size-w%self.patch_size
        else:
            pad_w = 0
        coord_pad = F.pad(hr_coord.permute(2, 0, 1), (0, pad_w, 0, pad_h), "constant", 0)
        coord_unfold = coord_pad.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        coord_patch_center = coord_unfold[:, :, :, self.patch_size//2, self.patch_size//2].permute(1, 2, 0)

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': coord_patch_center,
            'cell': cell,
            'gt': hr_rgb,   # for perceptual loss
            'gt_patch': hr_rgb_patch_center,    # for training: [-1, 1], residual
            'gt_lr_up': lr_up_patch_center,    # for training: [-1, 1], residual
            'interpolate_coord': hr_coord
        }    
@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
