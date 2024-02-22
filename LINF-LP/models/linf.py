import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import numpy as np

@register('linf')
class LINF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, flow_layers=10, num_layer=3, hidden_dim=256):
        super().__init__()        
        self.encoder = models.make(encoder_spec)

        # Fourier prediction
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1) # coefficient
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1) # frequency
        self.phase = nn.Linear(2, hidden_dim//2, bias=False) # phase 
        
        layers = []
        layers.append(nn.Conv2d(hidden_dim*4, hidden_dim, 1))
        layers.append(nn.ReLU())

        for i in range(num_layer-1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(hidden_dim, flow_layers*3*2, 1))
        self.layers = nn.Sequential(*layers)

        self.imnet = models.make(imnet_spec, args={'flow_layers': flow_layers})

    def gen_feat(self, inp):
        feat = self.encoder(inp)
        return feat

    def query_log_p(self, inp, feat, coord, cell, gt):
        # residual calculate in wrapper
        coef = self.coef(feat) # coefficient
        freq = self.freq(feat) # frequency

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # (48, 48, 2) -> (2, 48, 48) -> (1, 2, 48, 48) -> (16, 2, 48, 48)
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        freqs = []
        coefs = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]    # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_coord[:, 1, :, :] *= feat.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2] # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_cell[:, 1] *= feat.shape[-1]

                coef_ = F.grid_sample(coef, coord_.flip(-1), mode='nearest', align_corners=False)
                freq_ = F.grid_sample(freq, coord_.flip(-1), mode='nearest', align_corners=False)

                # basis
                freq_ = torch.stack(torch.split(freq_, freq.shape[1]//2, dim=1), dim=2)
                freq_ = torch.mul(freq_, rel_coord.unsqueeze(1))
                freq_ = torch.sum(freq_, dim=2)
                freq_ += self.phase(rel_cell).unsqueeze(-1).unsqueeze(-1)
                freq_ = torch.cat((torch.cos(np.pi*freq_), torch.sin(np.pi*freq_)), dim=1)

                freqs.append(freq_)
                coefs.append(coef_)

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # weighted coefficeint & apply coefficeint to basis
        for i in range(4):
            w = (areas[i]/tot_area).unsqueeze(1)
            coefs[i] = torch.mul(w*coefs[i], freqs[i])

        # concat fourier features of 4 LR pixels
        features = torch.cat(coefs, dim=1)

        # shared MLP
        affine_info = self.layers(features)

        # flow
        bs, w, h, _ = coord.shape
        z, log_p = self.imnet(gt.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1), affine_info.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1))
        z_reshape = z.reshape(bs, w, h, -1).permute(0, 3, 1, 2)

        return log_p, z_reshape
    
    def query_rgb(self, inp, feat, coord, cell, temperature=0, zmap=None):
        coef = self.coef(feat) # coefficient
        freq = self.freq(feat) # frequency

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # (48, 48, 2) -> (2, 48, 48) -> (1, 2, 48, 48) -> (16, 2, 48, 48)
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        freqs = []
        coefs = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]    # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                # coefficient & frequency prediction
                coef_ = F.grid_sample(coef, coord_.flip(-1), mode='nearest', align_corners=False)
                freq_ = F.grid_sample(freq, coord_.flip(-1), mode='nearest', align_corners=False)
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2] # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_cell[:, 1] *= feat.shape[-1]

                # basis
                freq_ = torch.stack(torch.split(freq_, freq.shape[1]//2, dim=1), dim=2)
                freq_ = torch.mul(freq_, rel_coord.unsqueeze(1))
                freq_ = torch.sum(freq_, dim=2)
                freq_ += self.phase(rel_cell).unsqueeze(-1).unsqueeze(-1)
                freq_ = torch.cat((torch.cos(np.pi*freq_), torch.sin(np.pi*freq_)), dim=1)

                freqs.append(freq_)
                coefs.append(coef_)

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # weighted coefficeint & apply coefficeint to basis
        for i in range(4):
            w = (areas[i]/tot_area).unsqueeze(1)
            coefs[i] = torch.mul(w*coefs[i], freqs[i])

        # concat fourier features of 4 LR pixels
        features = torch.cat(coefs, dim=1)

        # shared MLP
        affine_info = self.layers(features)

        # flow
        bs, w, h, _ = coord.shape
        if zmap is not None:
            pred = self.imnet.inverse(zmap.permute(0, 2, 3, 1).contiguous().view(-1, 3), affine_info.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1))
        else:
            pred = self.imnet.inverse((torch.randn((bs * w * h, 3)).cuda())*temperature, affine_info.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1))
        pred = pred.clone().view(bs, w, h, -1).permute(0, 3, 1, 2).contiguous()

        pred += F.grid_sample(inp, coord.flip(-1), mode='bilinear',\
                            padding_mode='border', align_corners=False)
        return pred

    def log_p(self, inp, coord, cell, gt):
        feat = self.gen_feat(inp)
        return self.query_log_p(inp, feat, coord, cell, gt)

    def rgb(self, inp, coord, cell, temperature=0, zmap=None):
        feat = self.gen_feat(inp)
        return self.query_rgb(inp, feat, coord, cell, temperature, zmap)

    def forward(self, op, inp=None, feat=None, coord=None, cell=None, gt=None, temperature=0, zmap=None):
        # op: "query_log_p", "query_rgb", "log_p", "rgb", "gen_feat"
        if op == "query_log_p":
            return self.query_log_p(inp, feat, coord, cell, gt)
        if op == "query_rgb":
            return self.query_rgb(inp, feat, coord, cell, temperature, zmap)
        if op == "log_p":
            return self.log_p(inp, coord, cell, gt)
        if op == "rgb":
            return self.rgb(inp, coord, cell, temperature, zmap)
        if op == "gen_feat":
            return self.gen_feat(inp)

@register('linf-patch')
class LINFPatch(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, flow_layers=10, num_layer=3, hidden_dim=256, patch_size=3):
        super().__init__()        
        self.patch_size = patch_size
        
        self.encoder = models.make(encoder_spec)
        # Fourier prediction
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1) # coefficient
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1) # frequency
        self.phase = nn.Linear(2, hidden_dim//2, bias=False) # phase 
        
        layers = []
        layers.append(nn.Conv2d(hidden_dim*4, hidden_dim, 1))
        layers.append(nn.ReLU())

        for i in range(num_layer-1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(hidden_dim, flow_layers*patch_size*patch_size*3*2, 1))
        self.layers = nn.Sequential(*layers)

        self.imnet = models.make(imnet_spec, args={'flow_layers': flow_layers, 'patch_size': patch_size})

    def gen_feat(self, inp):
        feat = self.encoder(inp)
        return feat

    def query_log_p(self, inp, feat, coord, cell, gt):
        # residual calculate in wrapper
        coef = self.coef(feat) # coefficient
        freq = self.freq(feat) # frequency

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # (48, 48, 2) -> (2, 48, 48) -> (1, 2, 48, 48) -> (16, 2, 48, 48)
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        freqs = []
        coefs = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]    # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_coord[:, 1, :, :] *= feat.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2] # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_cell[:, 1] *= feat.shape[-1]

                coef_ = F.grid_sample(coef, coord_.flip(-1), mode='nearest', align_corners=False)
                freq_ = F.grid_sample(freq, coord_.flip(-1), mode='nearest', align_corners=False)

                # basis
                freq_ = torch.stack(torch.split(freq_, freq.shape[1]//2, dim=1), dim=2)
                freq_ = torch.mul(freq_, rel_coord.unsqueeze(1))
                freq_ = torch.sum(freq_, dim=2)
                freq_ += self.phase(rel_cell).unsqueeze(-1).unsqueeze(-1)
                freq_ = torch.cat((torch.cos(np.pi*freq_), torch.sin(np.pi*freq_)), dim=1)

                freqs.append(freq_)
                coefs.append(coef_)

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # weighted coefficeint & apply coefficeint to basis
        for i in range(4):
            w = (areas[i]/tot_area).unsqueeze(1)
            coefs[i] = torch.mul(w*coefs[i], freqs[i])

        # concat fourier features of 4 LR pixels
        features = torch.cat(coefs, dim=1)

        # shared MLP
        affine_info = self.layers(features)

        # flow
        bs, w, h, _ = coord.shape
        z, log_p = self.imnet(gt.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1), affine_info.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1))
        z_reshape = z.reshape(bs, w, h, -1).permute(0, 3, 1, 2)

        return log_p, z_reshape
    
    def query_rgb(self, inp, feat, coord, cell, temperature=0, zmap=None):  
        coef = self.coef(feat) # coefficient
        freq = self.freq(feat) # frequency

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # (48, 48, 2) -> (2, 48, 48) -> (1, 2, 48, 48) -> (16, 2, 48, 48)
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        freqs = []
        coefs = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]    # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                # coefficient & frequency prediction
                coef_ = F.grid_sample(coef, coord_.flip(-1), mode='nearest', align_corners=False)
                freq_ = F.grid_sample(freq, coord_.flip(-1), mode='nearest', align_corners=False)
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2] # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_cell[:, 1] *= feat.shape[-1]

                # basis
                freq_ = torch.stack(torch.split(freq_, freq.shape[1]//2, dim=1), dim=2)
                freq_ = torch.mul(freq_, rel_coord.unsqueeze(1))
                freq_ = torch.sum(freq_, dim=2)
                freq_ += self.phase(rel_cell).unsqueeze(-1).unsqueeze(-1)
                freq_ = torch.cat((torch.cos(np.pi*freq_), torch.sin(np.pi*freq_)), dim=1)

                freqs.append(freq_)
                coefs.append(coef_)

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # weighted coefficeint & apply coefficeint to basis
        for i in range(4):
            w = (areas[i]/tot_area).unsqueeze(1)
            coefs[i] = torch.mul(w*coefs[i], freqs[i])

        # concat fourier features of 4 LR pixels
        features = torch.cat(coefs, dim=1)

        # shared MLP
        affine_info = self.layers(features)

        # flow
        bs, w, h, _ = coord.shape
        if zmap is not None:
            pred = self.imnet.inverse(zmap.permute(0, 2, 3, 1).contiguous().view(-1, 3*self.patch_size*self.patch_size), affine_info.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1))
        else:
            pred = self.imnet.inverse((torch.randn((bs * w * h, 3*self.patch_size*self.patch_size)).cuda())*temperature, affine_info.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1))
        pred = pred.clone().view(bs, w, h, -1).permute(0, 3, 1, 2).contiguous()
        
        pred = torch.nn.functional.fold(
            pred.view(bs, self.patch_size*self.patch_size*3, -1),
            output_size=(pred.shape[2]*self.patch_size, pred.shape[3]*self.patch_size),
            kernel_size=(self.patch_size, self.patch_size),
            stride=self.patch_size
        )
        return pred

    def log_p(self, inp, coord, cell, gt):
        feat = self.gen_feat(inp)
        return self.query_log_p(inp, feat, coord, cell, gt)

    def rgb(self, inp, coord, cell, temperature=0, zmap=None):
        feat = self.gen_feat(inp)
        return self.query_rgb(inp, feat, coord, cell, temperature, zmap)

    def forward(self, op, inp=None, feat=None, coord=None, cell=None, gt=None, temperature=0, zmap=None):
        # op: "query_log_p", "query_rgb", "log_p", "rgb", "gen_feat"
        if op == "query_log_p":
            return self.query_log_p(inp, feat, coord, cell, gt)
        if op == "query_rgb":
            return self.query_rgb(inp, feat, coord, cell, temperature, zmap)
        if op == "log_p":
            return self.log_p(inp, coord, cell, gt)
        if op == "rgb":
            return self.rgb(inp, coord, cell, temperature, zmap)
        if op == "gen_feat":
            return self.gen_feat(inp)