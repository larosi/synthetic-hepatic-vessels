# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:24:55 2024

@author: Mico
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import h5py


def walk_dataset(dataset_dir):
    """ walk image directory """
    all_full_paths = []
    all_roots = []
    all_names = []
    for root, dirs, files in os.walk(dataset_dir, topdown=True):
        for name in files:
            if '.nii' in name:
                all_full_paths.append(os.path.join(root, name))
                all_roots.append(root)
                all_names.append(name)
    return all_full_paths, all_roots, all_names

def split_per_patient(df, n_splits=5, patient_column='patient_id'):
    """ Create kfold splits from a dataframe using the patient_id """
    patient_ids = df[patient_column].unique()

    kf = KFold(n_splits=n_splits, random_state=44, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(patient_ids)):
        test_patients = patient_ids[test_index]
        df[f'fold_{i}'] = np.where(df[patient_column].isin(test_patients), 'test', 'train')
    return df

def create_df(dataset_dir, n_splits=5, masks_ref_name='segmentation'):
    """ walk a directory and create a df """
    all_full_paths, all_roots, all_names = walk_dataset(dataset_dir)
    df = pd.DataFrame()
    df['path'] = [abs_path.split(dataset_dir)[-1][1:] for abs_path in all_full_paths]
    df['patient_id'] = all_names
    df['patient_id'] = df['patient_id'].str.split('-').str[-1]
    df['patient_id'] = df['patient_id'].str.split('.nii').str[0]
    df.sort_values(['patient_id', 'path'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['is_mask'] = df['path'].str.lower().str.contains(masks_ref_name)
    df_mask = df[df['is_mask']].reset_index(drop=True)
    df = df[~df['is_mask']].reset_index(drop=True)
    df.drop(columns=['is_mask'], inplace=True)

    df.rename(columns={'path': 'volume_path'}, inplace=True)
    df['mask_path'] = df_mask['path']

    df.reset_index(drop=True, inplace=True)
    df = split_per_patient(df, n_splits=5)
    return df

def save_ct_images(filename, images, masks, images_fake, images_inpaint, patient_id, modality=None):
    """ Save list of images and mask in a .hdf5 dataset file

    Args:
        filename (str): dataset.hdf5 path.
        images (list(np.array)): list of images of each slice.
        masks (list(np.array)): list masks of each slice.
        patient_id (str): id of the patient.

    """
    idm = f'{patient_id}' if modality is None else f'{patient_id}_{modality}' 

    with h5py.File(filename, 'a') as h5f:
        if idm in h5f:
            print(f'{patient_id} already exists, it will be overwritten')
            del h5f[idm]
        patient_group = h5f.create_group(idm)
        for i, (image, mask, image_fake, image_inpaint) in enumerate(zip(images, masks, images_fake, images_inpaint)):
            patient_group.create_dataset(f'image/{i}',data=image,
                                         chunks=image.shape, compression="lzf")
            patient_group.create_dataset(f'mask/{i}', data=mask,
                                         chunks=mask.shape, compression="lzf")
            patient_group.create_dataset(f'image_fake/{i}',data=image_fake,
                                         chunks=image_fake.shape, compression="lzf")
            patient_group.create_dataset(f'image_inpaint/{i}',data=image_inpaint,
                                         chunks=image_inpaint.shape, compression="lzf")

def save_ct(ct_path, data, spatial_res):
    affine = np.diag(spatial_res + (1,))
    img = nib.Nifti1Image(data, affine)
    img.to_filename(ct_path)

def load_ct(ct_path):
    ct = nib.load(ct_path)
    spatial_res = tuple(np.diag(ct.affine)[0:3])
    return ct.get_fdata(), spatial_res

def apply_window_ct(ct, width, level):
    """ Normalize CT image using a window in the HU scale

    Args:
        ct (np.array): ct image.
        width (int): window width in the HU scale.
        level (int): center of the windows in the HU scale.

    Returns:
        ct (np.array): Normalized image in a range 0-1.

    """
    ct_min_val, ct_max_val = windowing_ct(width, level)
    ct_range = ct_max_val - ct_min_val
    ct = (ct - ct_min_val) / ct_range
    ct = np.clip(ct, 0, 1)
    return ct

def windowing_ct(width, level):
    """Generate CT bounds

    Args:
        width (int): window width in the HU scale.
        level (int): center of the windows in the HU scale.

    Returns:
        lower_bound (int): lower CT value.
        upper_bound (int): upper CT value.

    reference values:
    chest
    - lungs W:1500 L:-600
    - mediastinum W:350 L:50

    abdomen
    - soft tissues W:400 L:50
    - liver W:150 L:30

    spine
    - soft tissues W:250 L:50
    - bone W:1800 L:400

    head and neck
    - brain W:80 L:40
    - subdural W:130-300 L:50-100
    - stroke W:8 L:32 or W:40 L:40
    - temporal bones W:2800 L:600 or W:4000 L:700
    - soft tissues: W:350–400 L:20–60

    source: https://radiopaedia.org/articles/windowing-ct
    """
    lower_bound = level - width/2
    upper_bound = level + width/2
    return lower_bound, upper_bound

def to_unit8(image):
    image_uint8 = (image * 255).astype(np.uint8)
    return image_uint8

if __name__ == "__main__":
    from skimage import io
    import os
    dataset_name = '3Dircadb'
    dataset_dir = r'D:\datasets\medseg\3Dircadb'
    dataset_filename = 'gan_dataset.hdf5'

    """
    dataset_name = 'medicaldecathlon'
    dataset_dir = r'D:\datasets\medseg\medicaldecathlon\Task08_HepaticVessel'
    """
    #dataset_dir = r'D:\datasets\medseg\colorectal_dataset'
    #dataset_name = 'colorectal'
    dataset_dir = r'D:\datasets\medseg\colorectal_dataset'
    apply_window = True
    images_dir = os.path.join(dataset_dir, 'images') # volumen CT
    vessels_dir = os.path.join(dataset_dir, 'vessels')
    inpainted_dir = os.path.join(dataset_dir, 'inpainted')
    liver_dir = os.path.join(dataset_dir, 'liver')  # mascara solo del higado
    porta_dir = os.path.join(dataset_dir, 'porta_vein')  # mascara de vena porta cercana al higado
    hepatic_dir = os.path.join(dataset_dir, 'hepatic_vein')  # mascara de vena cava cercana al higado
    for patient_fn in tqdm(os.listdir(images_dir)):
        patient_id = patient_fn.split('.')[0]
        images3D, voxel_res = load_ct(os.path.join(images_dir, patient_fn))
        images3D_inpaint, voxel_res = load_ct(os.path.join(inpainted_dir, patient_fn))

        porta, _ = load_ct(ct_path=os.path.join(porta_dir, patient_fn))
        hepatic, _ = load_ct(ct_path=os.path.join(hepatic_dir, patient_fn))
        liver, _ = load_ct(ct_path=os.path.join(liver_dir, patient_fn))

        masks3D = np.logical_and(np.logical_or(porta, hepatic), liver)

        if apply_window:
            images3D = apply_window_ct(images3D, width=400, level=40)
            images3D_inpaint = apply_window_ct(images3D_inpaint, width=400, level=40)

        slices_to_keep = (masks3D==1).max(axis=(0, 1))
        slices_indices = np.where(slices_to_keep)[0]

        images = []
        images_fake = []
        images_inpaint = []
        masks = []
        for slice_i in slices_indices:
            mask = masks3D[:, :, slice_i].astype(int)
            image = images3D[:, :, slice_i]
            vessel_color = np.percentile(image[mask>0], 70) #np.median(images3D[masks3D > 0])
            image_fake = images3D_inpaint[:, :, slice_i]
            images_inpaint.append(to_unit8(image_fake))
            image_fake[mask > 0] = vessel_color

            images.append(to_unit8(image))
            images_fake.append(to_unit8(image_fake))
            masks.append(mask)
        mid_slice = len(images) // 2
        plt.imshow(images[mid_slice], cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imshow(images_fake[mid_slice], cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imshow(masks[mid_slice])
        plt.show()
        save_ct_images(dataset_filename, images, masks, images_fake, images_inpaint, patient_id)
