# modified from: https://github.com/yinboc/liif

import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.swin_transformer import SwinTransformer

import datasets
import models
import utils
from test import eval_psnr

import numpy as np
import random
import cv2

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=4, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_prior_model():
    if config.get('resume_prior_model') is not None:
        if os.path.exists(config.get('resume_prior_model')):
            sv_file = torch.load(config['resume_prior_model'])
            model = models.make(sv_file['prior_model'], load_sd=True).cuda()
            optimizer = utils.make_optimizer(
                model.parameters(), sv_file['optimizer'], load_sd=True)
            epoch_start = sv_file['epoch'] + 1
            if config.get('multi_step_lr') is None:
                lr_scheduler = None
            else:
                lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
            print("resume training from epoch {}".format(epoch_start))
        else:
            raise NotImplementedError('The path of desired checkpoint does not exist.')
    else:
        model = models.make(config['prior_model']).cuda()
        epoch_start = 1
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('Latent module: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def prepare_linf():
    if config.get('resume') is not None:
        if os.path.exists(config.get('resume')):
            sv_file = torch.load(config['resume'])
            model = models.make(sv_file['model'], load_sd=True).cuda()
        else:
            raise NotImplementedError('The path of desired checkpoint does not exist.')
    else:
        model = models.make(config['model']).cuda()

    log('LINF model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model

def train(train_loader, model, linf_model, optimizer, epoch, patch=False):
    model.train()
    linf_model.eval()

    vgg_loss = utils.Averager()
    latent_loss = utils.Averager()

    vgg_weight = config['loss_weight']['vgg']
    latent_weight = config['loss_weight']['latent']

    l1 = nn.L1Loss(reduction='mean')

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    num_dataset = 800 # DIV2K
    iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
                        * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div  # gt: original image, gt_patch: residual

        feat = linf_model("gen_feat", inp=inp)
        feat_linf = feat.detach()

        if patch:
            _, z_lr = linf_model("query_log_p", inp=inp, feat=feat_linf, coord=batch['coord'], cell=batch['cell'], gt=batch['gt_lr_up'])
            if latent_weight > 0:
                _, z_hr = linf_model("query_log_p", inp=inp, feat=feat_linf, coord=batch['coord'], cell=batch['cell'], gt=batch['gt_patch'])
        else:
            _, z_lr = linf_model("query_log_p", inp=inp, feat=feat_linf, coord=batch['coord'], cell=batch['cell'], gt=batch['gt_lr_up'])
            if latent_weight > 0:
                _, z_hr = linf_model("query_log_p", inp=inp, feat=feat_linf, coord=batch['coord'], cell=batch['cell'], gt=batch['gt_pixel'])

        z_lr = z_lr.detach().contiguous()
        z_lr_learned = model(z_lr, inp)

        """
        TODO: calculate loss
        """
        loss = 0

        if latent_weight > 0:
            latent_l = l1(z_lr_learned, z_hr.detach())
            latent_loss.add(latent_l.item())
        else:
            latent_l = 0.0

        if vgg_weight > 0:
            pred = linf_model("query_rgb", inp=inp, feat=feat, coord=batch['coord'], cell=batch['cell'], zmap=z_lr_learned)
            if patch:
                pred_vgg = pred + F.grid_sample(inp, batch['interpolate_coord'].flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
                vgg_l = l1(vgg(torch.clamp(pred_vgg * gt_div + gt_sub, 0, 1)), vgg(batch['gt']))
            else:
                vgg_l = l1(vgg(torch.clamp(pred * gt_div + gt_sub, 0, 1)), vgg(batch['gt']))
            vgg_loss.add(vgg_l.item())
        else:
            vgg_l = 0.0
            
        loss = vgg_l * vgg_weight + latent_l * latent_weight
        
        writer.add_scalars('loss', {'vgg_loss': vgg_l, 'latent_loss': latent_l}, (epoch-1)*iter_per_epoch + iteration)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ret_loss = []
        ret_loss.extend((vgg_loss.item(), latent_loss.item()))
        
        iteration += 1

        loss = None; vgg_l = None; latent_l = None
        
    return ret_loss


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    # Prepare linf model
    linf_model = prepare_linf()
    for param in linf_model.parameters():
        param.requires_grad = False

    # Build prior model
    in_chans = 3 * config['patch'] ** 2 if config.get('patch') else 3
    model, optimizer, epoch_start, lr_scheduler = prepare_prior_model()

    # Data parallel
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        class MyDataParallel(nn.DataParallel):
            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)
        model = MyDataParallel(model)   # able to access custom methods

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    timer = utils.Timer()

    print("lr: {}".format(optimizer.param_groups[0]['lr']))

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, linf_model, optimizer, epoch, config['patch'])
        log_info.append("train: vgg_l={:.4f}, latent_l: {:.4f}".format(train_loss[0], train_loss[1]))

        if lr_scheduler is not None:
            lr_scheduler.step()

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        model_spec = config['prior_model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'prior_model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'prior_model-epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'prior_model-epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model

            # TODO: complete test code
            val_res = eval_psnr(val_loader, linf_model, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                patch=config['patch'])

            log_info.append('val: psnr={:.4f}'.format(val_res))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--patch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6666)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("seed: {}".format(args.seed))
    config['seed'] = args.seed

    torch.backends.cudnn.benchmark = True

    vgg_weight = config['loss_weight']['vgg']
    latent_weight = config['loss_weight']['latent']
    print("vgg weight: {}, latent_weight: {}".format(vgg_weight, latent_weight))

    if vgg_weight > 0:
        vgg = models.make({'name': 'VGGFeatureExtractor', 'args': {'feature_layer': 34, 'use_bn': False}}).cuda()
    config['train_dataset']['wrapper']['name'] += '-crop' # all use crop dataset

    config['patch'] = args.patch
    if args.patch:
        assert args.patch > 1 and args.patch % 2 == 1   # patch size must be odd number greater than one
        config['train_dataset']['wrapper']['name'] += '-patch'
        config['train_dataset']['wrapper']['args']['patch_size'] = args.patch
        config['val_dataset']['wrapper']['name'] += '-patch'
        config['val_dataset']['wrapper']['args']['patch_size'] = args.patch
        config['model']['name'] += '-patch'
        config['model']['args']['patch_size'] = args.patch
        config['prior_model']['args']['in_chans'] *= (args.patch ** 2)

    print("train dataset: {}".format(config['train_dataset']['wrapper']['name']))
    print("val dataset: {}".format(config['val_dataset']['wrapper']['name']))

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
