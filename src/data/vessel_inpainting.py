# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:13:07 2024

@author: Mico
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import cv2


def load_ct(ct_path):
    ct = nib.load(ct_path)
    spatial_res = tuple(np.diag(ct.affine)[0:3])
    ct = ct.get_fdata()
    return ct, spatial_res

def save_ct(ct_path, data, spatial_res):
    affine = np.diag(spatial_res + (1,))
    img = nib.Nifti1Image(data, affine)
    img.to_filename(ct_path)

dataset_dir = r'D:\datasets\medseg\colorectal_dataset'
#dataset_dir = r'D:\datasets\medseg\3Dircadb'
images_dir = os.path.join(dataset_dir, 'images') # volumen CT
segmentation_dir = os.path.join(dataset_dir, 'segmentations') # mascaras de todas las clases en un solo archivo
liver_dir = os.path.join(dataset_dir, 'liver')  # mascara solo del higado
porta_dir = os.path.join(dataset_dir, 'porta_vein')  # mascara de vena porta cercana al higado
hepatic_dir = os.path.join(dataset_dir, 'hepatic_vein')  # mascara de vena cava cercana al higado
pred_dir = os.path.join(dataset_dir, 'pred')

vessels_dir = os.path.join(dataset_dir, 'vessels')
inpainted_dir = os.path.join(dataset_dir, 'inpainted')
os.makedirs(vessels_dir, exist_ok=True)
os.makedirs(inpainted_dir, exist_ok=True)


filenames = os.listdir(images_dir)  # lista de pacientes

for patient_fn in tqdm(filenames):
    
    #patient_fn = np.random.choice(filenames)
    #patient_fn

    porta, _ = load_ct(ct_path=os.path.join(porta_dir, patient_fn))
    hepatic, _ = load_ct(ct_path=os.path.join(hepatic_dir, patient_fn))
    liver, _ = load_ct(ct_path=os.path.join(liver_dir, patient_fn))
    pred, _ = load_ct(ct_path=os.path.join(pred_dir, patient_fn))
    
    masks = np.logical_and(np.logical_or(np.logical_or(porta, hepatic), pred), liver)
    masks_dilated = np.logical_and(binary_dilation(masks, iterations=2), liver)
    ct, spatial_res = load_ct(ct_path=os.path.join(images_dir, patient_fn))
    
    # soft tissue ct window https://radiopaedia.org/articles/windowing-ct
    ct_min = -160
    ct_max = 240
    ct = (ct - ct_min) / (ct_max - ct_min)
    ct = np.clip(ct, 0, 1)
    ct = np.uint8(ct*255)

    ct_inpainted = []
    for slice_i in range(0, ct.shape[-1]):
        image_defect = ct[:,:,slice_i:slice_i+1]
        mask = masks_dilated[:,:,slice_i:slice_i+1]
        
        image_result = image_defect.copy()
        mask1 = cv2.bitwise_not(np.uint8(mask*255))
    
        cv2.xphoto.inpaint(image_defect, 
                           mask1,
                           image_result, cv2.xphoto.INPAINT_FSR_FAST)
        ct_inpainted.append(image_result)
    
    ct_inpainted = np.dstack(ct_inpainted)

    ct_inpainted = ((ct_inpainted / 255.0) * (ct_max - ct_min) + ct_min)

    ct_inpainted_path = os.path.join(inpainted_dir, patient_fn)
    vessels_path = os.path.join(vessels_dir, patient_fn)
    
    save_ct(vessels_path, masks.astype(int), spatial_res)
    save_ct(ct_inpainted_path, ct_inpainted, spatial_res)

    slice_i = ct.shape[-1] // 2
    plt.imshow(ct[:,:,slice_i])
    plt.show()
    plt.imshow(ct_inpainted[:,:,slice_i])
    plt.show()
    plt.imshow(masks[:,:,slice_i])
    plt.show()
    plt.imshow(masks_dilated[:,:,slice_i])
    plt.show()
    
    
    
    
    
