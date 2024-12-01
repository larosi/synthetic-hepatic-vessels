# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:41:17 2024

@author: Mico
"""

import torch
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

from smdino.medical_sdino import load_sdino, PatchEmbed
from smdino.modeling.decoder import DecoderTokens

def load_ct(ct_path):
    ct = nib.load(ct_path)
    spatial_res = tuple(np.diag(ct.affine)[0:3])
    ct = ct.get_fdata()
    return ct, spatial_res

def save_ct(ct_path, data, spatial_res):
    affine = np.diag(spatial_res + (1,))
    img = nib.Nifti1Image(data, affine)
    img.to_filename(ct_path)


class AxialCTDataset(Dataset):
    def __init__(self, ct, img_size=448):
      mean = (0.5,)
      std = (0.5,)
      self.ct = ct
      self.img_size = img_size
      self.image_transforms = T.Compose([T.Resize((self.img_size, self.img_size)),
                              T.CenterCrop((self.img_size, self.img_size)),
                              T.ToTensor(),
                              T.Normalize(mean, std)])
    def __len__(self):
        return self.ct.shape[-1]

    def __getitem__(self, idx):
        img = self.ct[:, :, idx]
        img = Image.fromarray(img)
        img = self.image_transforms(img)
        return img

encoder_path = r'C:\Users\Mico\Desktop\github\uandes\steerable-medical-dino\models\medDinoDIET_CSM.pth' 
decoder_path =  r'C:\Users\Mico\Desktop\github\uandes\liver-vessel-segmentation\models\fold_1_vessel_segmentor.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
encoder.patch_embed.proj = PatchEmbed()
encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
encoder.to(device)

decoder = DecoderTokens(in_ch=768, out_ch=1,
                        transformer_dim=256,
                        image_size=512,
                        num_registers=5, use_input_conv=True)
decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
decoder.to(device)

print(encoder)
print(decoder)


dataset_dir = r'D:\datasets\medseg\3Dircadb'
# dataset_dir = r'D:\datasets\medseg\colorectal_dataset'

images_dir = os.path.join(dataset_dir, 'images') # volumen CT
pred_dir = os.path.join(dataset_dir, 'pred')
os.makedirs(pred_dir, exist_ok=True)


for patient_fn in tqdm(os.listdir(images_dir)):
    ct_path = os.path.join(images_dir, patient_fn)
    ct, spatial_res = load_ct(ct_path)

    # soft tissue ct window https://radiopaedia.org/articles/windowing-ct
    ct_min = -160
    ct_max = 240
    ct = (ct - ct_min) / (ct_max - ct_min)
    ct = np.clip(ct, 0, 1)
    ct = np.uint8(ct*255)
    
    
    ct_dataset = AxialCTDataset(ct, img_size=518)
    ct_loader = DataLoader(ct_dataset, batch_size=16, shuffle=False)
    predictions = []
    encoder.eval()
    decoder.eval()
    for images in ct_loader:
        with torch.no_grad():
            images = images.to(device)
            features = encoder.forward_features(images)
            masks_pred = decoder(features['x_norm_patchtokens'].detach())
            masks_pred = torch.nn.functional.sigmoid(masks_pred)
    
            predictions.append(masks_pred.cpu().detach().numpy())
    predictions = np.vstack(predictions)
    predictions = (predictions >= 0.5) # .astype(int)
    predictions = np.squeeze(np.moveaxis(predictions, 0, -1))
    
    # save predictions
    pred_path = os.path.join(pred_dir, patient_fn)
    save_ct(pred_path, predictions.astype(int), spatial_res)



