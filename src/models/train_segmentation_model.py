# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:34:26 2024

@author: Mico
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
import imageio

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model_arch import AutoEncoder
from dataset_generator import SegmentationDataset
from dataset_usage_example import create_df_slices, split_per_patient

import monai


if __name__ == "__main__":

    # https://drive.google.com/file/d/1lZFvA6dhzZ01Un08DDLwDdPCtse3iRGa/view?usp=sharing
    dataset_dir = os.path.join('data')
    hdf5_path = os.path.join(dataset_dir, 'real_and_synthetic_dataset.hdf5')

    df = create_df_slices(hdf5_path)
    df = df[df['dataset'] != '3Dircadb'] # usamos solo colorectal
    df['is_synthetic'] = df['patient_id'].str.contains('synthetic_')
    df['original_patient_id'] = df['patient_id'].str.split('_').str[-1]
    
    number_of_folds = 5
    df = split_per_patient(df, n_splits=number_of_folds, patient_column='original_patient_id')
    
    for fold in range(0, number_of_folds):
        # los datos de test solo pueden ser reales para que los experimentos sean comparables entre si
        df.loc[np.logical_and(df[f'fold_{fold}']=='test', df['is_synthetic']), f'fold_{fold}'] = 'val' 
    df.to_csv(os.path.join('data', 'patient_list.csv'), index=False)

    experiment_names = ['real_and_synthetic', 'real', 'synthetic'] # experimentos 1:1 1:0 0:1
    experiment_name = experiment_names[2] # elegir manualmente el experimento a correr


    for fold in tqdm(np.arange(0, number_of_folds)[::-1]):
        df_train = df[df[f'fold_{fold}'] == 'train'].reset_index(drop=True)
        df_test = df[df[f'fold_{fold}'] == 'test'].reset_index(drop=True)
        
        if experiment_name == 'synthetic':
            df_train = df_train[df_train['is_synthetic']]
        elif experiment_name == 'real':
            df_train = df_train[~df_train['is_synthetic']]
        df_train.reset_index(inplace=True, drop=True)
    
        input_size = 512
        train_dataset = SegmentationDataset(df_train, hdf5_path,
                                   img_size=input_size,
                                   mask_size=input_size)
    
        test_dataset = SegmentationDataset(df_test, hdf5_path,
                                  img_size=input_size,
                                  mask_size=input_size)
    
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        model = AutoEncoder(input_size=input_size,
                            channels=[64, 128, 256])
        model = model.to(device)
    
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print('Number of parameters: %d' % num_params)
    
        patience = 50
        num_epochs = 100
        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=learning_rate,
                                      weight_decay=0.01,
                                      amsgrad=False)
    
        criterion = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=num_epochs,
                                                               eta_min=learning_rate/10)
    
        desired_metrics = ['loss']
        splits = ['train', 'test']
        history = {'epoch': []}
        for metric_name in desired_metrics:
            for split in splits:
                history[f'{split}_{metric_name}'] = []
    
        display_images = True
        with tqdm(total=num_epochs, desc='epoch', position=0) as pbar:
            for epoch in range(0, num_epochs):  
                total_train_loss = 0
                model.train()
                optimizer.zero_grad()
    
                for images, masks in tqdm(train_loader, position=1, leave=False, desc='train batch'):
                    masks = masks.to(device)
                    masks_pred, latent = model(images.to(device))
    
                    loss = criterion(masks_pred, masks)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
    
                scheduler.step()
    
                # test loop
                total_test_loss = 0
                model.eval()
                with torch.no_grad():
                    for images, masks in tqdm(test_loader, position=1, leave=False, desc='test batch'):
                        masks = masks.to(device)
                        masks_pred, latent = model(images.to(device))
           
                        loss = criterion(masks_pred, masks)
                        total_test_loss += loss.item()
    
    
                avg_train_loss = total_train_loss / len(train_loader)
                avg_test_loss = total_test_loss / len(test_loader)
    
                # metric tracking
                history['epoch'].append(epoch)
                history['train_loss'].append(avg_train_loss)
                history['test_loss'].append(avg_test_loss)
    
                # early stopping
                df_history = pd.DataFrame(history)
                df_history['is_improvement'] = df_history['test_loss'] <= df_history['test_loss'].min()
                epochs_since_improvement = epoch - df_history.iloc[df_history['is_improvement'].argmax()]['epoch']
                if epochs_since_improvement == 0:
                    torch.save(model.state_dict(), os.path.join('models', f'{experiment_name}_kfold_{fold}_best_model.pth'))
                    df_history.to_csv(os.path.join('models', f'{experiment_name}_kfold_{fold}_training_history.csv'))
                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
    
                #if epoch % 10 == 0:
                if epoch > -1:
                    io.imshow(images[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
                    io.show()
        
                    io.imshow(masks_pred[0, 0, :, :].cpu().detach().numpy() > 0, cmap='gray')
                    io.show()
        
        
                    io.imshow(masks[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
                    io.show()
    
                    df_history[['train_loss', 'test_loss']].plot()
                    io.show()
    
        df_history[['train_loss', 'test_loss']].plot()
        io.show()
