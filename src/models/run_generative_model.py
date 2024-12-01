# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:56:06 2024

@author: Mico
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_ubyte
import imageio

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model_arch import AutoEncoder

from create_gan_dataset import save_ct_images
from dataset_generator import GanDataset, NewMaskDataset
from dataset_usage_example import dataframe_from_hdf5, split_per_patient, get_images

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_dilation


def elastic_transform(image, alpha, sigma, random_state=None, use_gray=True):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)        
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def generate_masks(mask_example, number_of_masks=5):
    new_masks = []
    single_ch = len(mask_example.shape) == 2
    if single_ch:
        mask_example = np.expand_dims(mask_example, axis=-1)
    for i in range(0, number_of_masks):
        mask_new = elastic_transform(mask_example, alpha=900, sigma=4)
        mask_new = binary_dilation(mask_new>0)
        if single_ch:
            mask_new = np.squeeze(mask_new)
        new_masks.append(mask_new)
    return new_masks



if __name__ == "__main__":

    # https://drive.google.com/file/d/1erq2DUnqtLXhI4vHB0oqb25rdzUjg86j/view?usp=sharing
    dataset_dir = os.path.join('data')
    hdf5_path = os.path.join(dataset_dir, 'gan_dataset.hdf5')
    out_hdf5_path = os.path.join(dataset_dir, 'generated_dataset.hdf5')

    df = dataframe_from_hdf5(hdf5_path)
    
    input_size = 512
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(input_size=input_size,
                        channels=[64, 128, 256])
    model_path = os.path.join('models', 'ae', 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    display_images = True

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        patient_id = row['patient_id']
        list_new_fakes = []
        list_new_masks = []
        images_inpaint = []
        for slice_id in range(0, row['slices']):
            img, mask, img_fake, img_inpaint = get_images(hdf5_path, patient_id, slice_id)
            new_masks = generate_masks(mask, number_of_masks=1)

            ref_color = int(np.mean(img_fake[mask>0]))
            for new_mask in new_masks:
                new_fake = np.expand_dims(img_inpaint.copy(), axis=-1)
                new_fake[new_mask>0] = ref_color # TODO: add color variance
                list_new_fakes.append(np.squeeze(new_fake))
                list_new_masks.append(new_mask)
                images_inpaint.append(img_inpaint)

        val_dataset = NewMaskDataset(list_new_fakes, list_new_masks)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if display_images:
            io.imshow(img)
            plt.title(f'{patient_id}\nReal')
            io.show()
    
        with torch.no_grad():
            model.eval()
            new_images = []
            new_masks = []
            for i, (images, masks) in enumerate(val_loader):
                #images = images.to(device)
                #masks = masks
                recon, latent = model(images.to(device))
                recon = F.sigmoid(recon)
       
                recon = recon[0, 0, :, :].detach().cpu().numpy()
                mask_recon = masks[0, 0, :, :].detach().cpu().numpy() > 0.5
                im_merged = images[0, 0, :, :].detach().cpu().numpy()
                im_merged[mask_recon>0] = recon[mask_recon>0]
                
                new_images.append(img_as_ubyte(im_merged))
                new_masks.append(mask_recon)
    
            if display_images:
                io.imshow(im_merged)
                plt.title(f'{patient_id}\nGenerated N° {i}')
                io.show()
            new_patient_id = f'synthetic_{patient_id}'
            save_ct_images(out_hdf5_path,
                           new_images,
                           new_masks,
                           list_new_fakes,
                           images_inpaint,
                           new_patient_id)
                

    

