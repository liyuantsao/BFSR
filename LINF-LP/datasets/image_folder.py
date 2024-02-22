import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    'bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('image-folder-DF2K')
class ImageFolderDF2K(Dataset):

    def __init__(self, root_path_D2K, root_path_F2K, first_k=None, repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        self.files = []

        print('loading DIV2K...')
        filenames = sorted(os.listdir(root_path_D2K))
        if first_k is not None:
            filenames = filenames[:first_k]
        for filename in filenames:
            file = os.path.join(root_path_D2K, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path_D2K),
                    'bin_' + os.path.basename(root_path_D2K))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

        # Flickr2K:
        print('loading Flickr2K...')
        filenames = sorted(os.listdir(root_path_F2K))
        if first_k is not None:
            filenames = filenames[:first_k]
        for i, filename in enumerate(filenames):
            file = os.path.join(root_path_F2K, filename)
            
            # if cache == 'none':
            #     self.files.append(file)

            # elif cache == 'bin':
            #     bin_root = os.path.join(os.path.dirname(root_path_F2K),
            #         'bin_' + os.path.basename(root_path_F2K))
            #     if not os.path.exists(bin_root):
            #         os.mkdir(bin_root)
            #         print('mkdir', bin_root)
            #     bin_file = os.path.join(
            #         bin_root, filename.split('.')[0] + '.pkl')
            #     if not os.path.exists(bin_file):
            #         with open(bin_file, 'wb') as f:
            #             pickle.dump(imageio.imread(file), f)
            #         print('dump', bin_file)
            #     self.files.append(bin_file)

            # elif cache == 'in_memory':  # we don't apply in_memory to Flickr2K
            #     bin_root = os.path.join(os.path.dirname(root_path_F2K),
            #         'bin_' + os.path.basename(root_path_F2K))
            #     if not os.path.exists(bin_root):
            #         os.mkdir(bin_root)
            #         print('mkdir', bin_root)
            #     bin_file = os.path.join(
            #         bin_root, filename.split('.')[0] + '.pkl')
            #     if not os.path.exists(bin_file):
            #         with open(bin_file, 'wb') as f:
            #             pickle.dump(imageio.imread(file), f)
            #         print('dump', bin_file)
            #     self.files.append(bin_file)
            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                self.files.append(file)

            elif cache == 'in_memory':
                self.files.append(file) # only apply in_memory to DIV2K

        print('finish loading {} files'.format(len(self.files)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            if type(x) == str:  # Flickr2K
                if x[-4:] != '.pkl':    # none for Flickr2K
                    x = transforms.ToTensor()(Image.open(x).convert('RGB'))
                else:   # bin for Flickr2K
                    with open(x, 'rb') as f:
                        x = pickle.load(f)
                    x = np.ascontiguousarray(x.transpose(2, 0, 1))
                    x = torch.from_numpy(x).float() / 255
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
