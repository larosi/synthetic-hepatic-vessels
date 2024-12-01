# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:18:28 2024

@author: Mico
"""
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def get_image_and_mask(hdf5_path, patient_id, slice_id):
    with h5py.File(hdf5_path, 'r') as h5f:
        idm = f'{patient_id}'
        img = h5f[f'{idm}/image/{slice_id}'][()]
        mask = h5f[f'{idm}/mask/{slice_id}'][()]

    return img, mask

def get_images(hdf5_path, patient_id, slice_id):
    with h5py.File(hdf5_path, 'r') as h5f:
        idm = f'{patient_id}'
        img = h5f[f'{idm}/image/{slice_id}'][()]
        mask = h5f[f'{idm}/mask/{slice_id}'][()]
        img_fake = h5f[f'{idm}/image_fake/{slice_id}'][()] 
        img_inpaint = h5f[f'{idm}/image_inpaint/{slice_id}'][()] 
        
    return img, mask, img_fake, img_inpaint

def dataframe_from_hdf5(hdf5_path):
    df = {'patient_id': [], 'dataset': [], 'slices': []}
    with h5py.File(hdf5_path, 'r') as h5f:
        patients = list(h5f.keys())
        for idm in patients:
            num_slices = max([int(k) for k in h5f[f'{idm}/image'].keys()])
            dataset = 'colorectal' if 'CRLM-' in idm else '3Dircadb'
            df['patient_id'].append(idm)
            df['dataset'].append(dataset)
            df['slices'].append(num_slices)
    df = pd.DataFrame(df)
    return df

def split_per_patient(df, n_splits=5, patient_column='patient_id'):
    """ Create kfold splits from a dataframe using the patient_id """
    patient_ids = df[patient_column].unique()

    kf = KFold(n_splits=n_splits, random_state=44, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(patient_ids)):
        test_patients = patient_ids[test_index]
        df[f'fold_{i}'] = np.where(df[patient_column].isin(test_patients), 'test', 'train')
    return df

def create_df_slices(hdf5_path):
    df = dataframe_from_hdf5(hdf5_path)
    df_slices = {'patient_id': [], 'dataset': [], 'slice_id': []}
    for index, row in df.iterrows():
        n_slices = row['slices']
        df_slices['patient_id'] += [row['patient_id']] * n_slices
        df_slices['dataset'] += [row['dataset']] * n_slices
        df_slices['slice_id'] += list(np.arange(0, n_slices))
    df_slices = pd.DataFrame(df_slices)
    return df_slices

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    # https://drive.google.com/file/d/1erq2DUnqtLXhI4vHB0oqb25rdzUjg86j/view?usp=sharing
    dataset_dir = r'.'
    hdf5_path = os.path.join(dataset_dir, 'gan_dataset.hdf5')
    df = dataframe_from_hdf5(hdf5_path)

    sample = df.sample(n=1).iloc[0]
    patient_id = sample['patient_id']
    dataset = sample['dataset']
    slice_id = np.random.randint(0, sample['slices'])
    img, mask, img_fake, img_inpaint = get_images(hdf5_path, patient_id, slice_id)

    img_mask = mark_boundaries(img, mask)

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Real Image')
    plt.show()

    plt.imshow(img_fake, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Fake Image')
    plt.show()
    
    plt.imshow(img_mask)
    plt.title(f'Image and Mask')
    plt.show()
    
    plt.imshow(img_inpaint, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Image Inpainted')
    plt.show()
