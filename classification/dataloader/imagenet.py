import logging
import os
from glob import glob
from os import path as osp

import mat4py
import numpy as np
import torch
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')

class IN1K(torch.utils.data.Dataset):
    """
    ImageNet 1K dataset
    Classes numbered from 0 to 999 inclusive
    """
    NUM_CLASSES = 1000
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    

    def __init__(self, root='/esat/visicsrodata/datasets/ilsvrc2012/', split='train', transform=None):
        print('Dataset: ImageNet')
        self.root = root
        self.transform = transform

        
        self.loader = default_loader
        self.synsets = mat4py.loadmat(osp.join(root,
                                               'ILSVRC2012_devkit_t12',
                                               'data', 
                                               'meta.mat'))['synsets']
        self.class_to_word = {wn: word
                            for (wn, word) in zip(self.synsets['WNID'],
                                                   self.synsets['words'])}
        self.classes = [wn for (wn, label) in zip(self.synsets['WNID'],
                                                   self.synsets['ILSVRC2012_ID'])
                                           if label <= self.NUM_CLASSES]
        self.classes.sort()
        self.class_to_idx = {c: i for (i, c) in enumerate(self.classes)}
        self.idx_to_class = {i: c for (i, c) in enumerate(self.classes)}
        self.ilsvrc_label_to_label = {label: self.class_to_idx[wn]
                                      for (wn, label) in zip(self.synsets['WNID'],
                                                             self.synsets['ILSVRC2012_ID'])
                                      if label <= self.NUM_CLASSES}

        if split == 'train':
            open(osp.join(osp.dirname(__file__), 'imagenet_cache.py'), 'a')
            import dataloader.imagenet_cache  as imagenet_cache
            if hasattr(imagenet_cache, 'train_cache'):
                self.imgs = imagenet_cache.train_cache
            else:
                self.imgs = self._rebuild_train()
            self.imgs.sort()
            self.labels = []
            for im in self.imgs:
                cls = osp.basename(osp.dirname(im))
                self.labels.append(self.class_to_idx[cls])

            
        elif split == 'val':
            self.imgs = [osp.relpath(f, root)
                         for f in glob(osp.join(root, 'ILSVRC2012_img_val', '*.JPEG'))]
            self.imgs.sort()
            ilsvrc_labels = np.loadtxt(osp.join(root,
                                                'ILSVRC2012_devkit_t12',
                                                'data',
                                                'ILSVRC2012_validation_ground_truth.txt'
                                                ), dtype=int)

            self.labels = [self.ilsvrc_label_to_label[label] for label in ilsvrc_labels]
            assert len(self.imgs) == len(self.labels)
            sort_by_label = sorted(zip(self.labels, self.imgs))
            self.labels, self.imgs = list(zip(*sort_by_label))
        
        else:
            raise ValueError(f'Split {split} unknown')
        
        assert len(self.imgs) == len(self.labels)
        print('Imagenet: number of images:', len(self.imgs))
    
    def _rebuild_train(self):
        """
        Crawls train images and caches them
        """
        import dataloader.imagenet_cache as imagenet_cache
        
        train_dir = osp.join(self.root, 'ILSVRC2012_img_train')
        logging.info(f'Rebuilding imagenet train cache in {train_dir}')
        self.imgs = [osp.relpath(f, self.root)
                for f in sorted(glob(osp.join(train_dir, '*', '*.JPEG')))]
        
        with open(imagenet_cache.__file__, 'w') as f:
            f.write('train_cache = ["' + '", \n"'.join(self.imgs) + '"]')
        return self.imgs
        
    def __getitem__(self, idx):
        image = self.loader(osp.join(self.root, self.imgs[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return (image, self.labels[idx])
    
    def __len__(self):
        return len(self.imgs)
    
    def __repr__(self):
        return f"IN1K(root='{self.root}', split='{self.split}')"
