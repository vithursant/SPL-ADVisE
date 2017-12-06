from __future__ import print_function
import numpy as np

import torch
from torch.utils import data

from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision.transforms import ToTensor

import pdb

class CUB2002011(data.Dataset):
    base_folder = 'CUB_200_2011'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    tarname = 'CUB_200_2011.tgz'
    dirname = 'CUB_200_2011'

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False,
                 target_size=[224, 224], nb_examples=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.base = os.path.join(self.root, self.dirname)

        if download:
            self.download()

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        from scipy import misc

        #to_tensor = ToTensor()

        filepaths = np.loadtxt(os.path.join(self.base, 'images.txt'),
                               dtype=bytes).astype(str)[:, 1]
        nb_examples = nb_examples or len(filepaths)
        imgs = torch.zeros(nb_examples, 3, *target_size)
        labs = torch.zeros(nb_examples).long()
        img_base = os.path.join(self.base, 'images')
        for i, fp in enumerate(filepaths):
            if i >= nb_examples:
                break
            img_path = os.path.join(img_base, fp)
            img = misc.imresize(misc.imread(img_path, mode="RGB"), target_size + [3])
            #pdb.set_trace()
            #imgs[i] = to_tensor(img)
            #pdb.set_trace()
            imgs[i] = torch.from_numpy(img)
            labs[i] = int(fp[:3])
        #pdb.set_trace()
        self.data = imgs
        #pdb.set_trace()
        self.labels = labs

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        tar_path = os.path.join(self.root, self.tarname)
        if check_integrity(tar_path, self.tgz_md5):
            print("Tar has been previously downloaded")
            return

        download_url(self.url, self.root, self.tarname, self.tgz_md5)

        print("Extracting Files")
        import tarfile
        base = os.path.join(self.root, self.dirname)
        tar_path = os.path.join(base, self.tarname)
        cwd = os.getcwd()
        tar = tarfile.open(tar_path, "r:gz")
        os.chdir(base)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
