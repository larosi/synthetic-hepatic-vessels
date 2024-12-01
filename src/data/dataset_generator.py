# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:41:13 2024

@author: Mico
"""

import os
import PIL
from random import random
from scipy.ndimage import binary_dilation

from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

from dataset_usage_example import get_images, dataframe_from_hdf5, get_image_and_mask

class NewMaskDataset(Dataset):
    def __init__(self, images, masks):
      self.images = images
      self.masks = masks
      self.img_size = 512
      self.transforms = T.Compose([T.Resize((self.img_size, self.img_size)),
                                   T.CenterCrop((self.img_size, self.img_size)),
                                   T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        img = PIL.Image.fromarray(img)
        mask = PIL.Image.fromarray(mask)
        img = self.transforms(img)
        mask = self.transforms(mask)
        return img, mask

class GanDataset(Dataset):
    def __init__(self, df, hdf5_path, img_size=512, mask_size=512, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.hdf5_path = hdf5_path
        self.img_size = img_size
        self.mask_size = mask_size

        self.image_transforms = T.Compose([T.Resize((self.img_size, self.img_size)),
                                T.CenterCrop((self.img_size, self.img_size)),
                                T.ToTensor()])

        self.mask_transforms = T.Compose([T.Resize((self.mask_size, self.mask_size)),
                                          T.CenterCrop((self.mask_size, self.mask_size)),
                                          T.ToTensor()])
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hdf5_path = self.hdf5_path
        patient_id = row['patient_id']
        slice_id = row['slice_id']
        img, mask, target, _ = get_images(hdf5_path, patient_id, slice_id)
        img = PIL.Image.fromarray(img)
        target = PIL.Image.fromarray(target)
        mask = binary_dilation(mask)
        mask = PIL.Image.fromarray(mask)

        # apply transforms and data augmentation
        img = self.image_transforms(img)
        target = self.image_transforms(target)
        mask = self.mask_transforms(mask) 

        return img, target, mask


class SegmentationDataset(Dataset):
    def __init__(self, df, hdf5_path, img_size=512, mask_size=512, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.hdf5_path = hdf5_path
        self.img_size = img_size
        self.mask_size = mask_size

        self.image_transforms = T.Compose([T.Resize((self.img_size, self.img_size)),
                                T.CenterCrop((self.img_size, self.img_size)),
                                T.ToTensor()])

        self.mask_transforms = T.Compose([T.Resize((self.mask_size, self.mask_size)),
                                          T.CenterCrop((self.mask_size, self.mask_size)),
                                          T.ToTensor()])
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hdf5_path = self.hdf5_path
        patient_id = row['patient_id']
        slice_id = row['slice_id']
        img, mask = get_image_and_mask(hdf5_path, patient_id, slice_id)
        img = PIL.Image.fromarray(img)
        mask = PIL.Image.fromarray(mask)

        # apply transforms and data augmentation
        img = self.image_transforms(img)
        mask = self.mask_transforms(mask) 

        return img, mask
    