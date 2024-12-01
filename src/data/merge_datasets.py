# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:34:21 2024

@author: Mico
"""

import os
from tqdm import tqdm
from create_gan_dataset import save_ct_images
from dataset_usage_example import dataframe_from_hdf5, get_images

dataset_dir = os.path.join('data')
hdf5_path = os.path.join(dataset_dir, 'gan_dataset.hdf5')
out_hdf5_path = os.path.join(dataset_dir, 'real_and_synthetic_dataset.hdf5')


df = dataframe_from_hdf5(hdf5_path)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    patient_id = row['patient_id']
    images = []
    masks = []
    images_fake = []
    images_inpaint = []
    for slice_id in range(0, row['slices']):
        img, mask, img_fake, img_inpaint = get_images(hdf5_path, patient_id, slice_id)
        images.append(img)
        masks.append(mask)
        images_fake.append(img_fake)
        images_inpaint.append(img_inpaint)

    save_ct_images(out_hdf5_path, images, masks, images_fake, images_inpaint, patient_id)