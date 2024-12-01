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
from dataset_generator import GanDataset
from dataset_usage_example import create_df_slices, split_per_patient


if __name__ == "__main__":

    # https://drive.google.com/file/d/1erq2DUnqtLXhI4vHB0oqb25rdzUjg86j/view?usp=sharing
    dataset_dir = os.path.join('data')
    hdf5_path = os.path.join(dataset_dir, 'gan_dataset.hdf5')

    df = create_df_slices(hdf5_path)
    #df = df[df['dataset'] == '3Dircadb']

    df = split_per_patient(df, n_splits=5, patient_column='patient_id')
    fold = 0

    df_train = df[df[f'fold_{fold}'] == 'train'].reset_index(drop=True)
    df_test = df[df[f'fold_{fold}'] == 'test'].reset_index(drop=True)
    
    #df_train = df_train.sample(n=2000).reset_index(drop=True)
    #df_test = df_test.sample(n=1000).reset_index(drop=True)

    input_size = 512
    train_dataset = GanDataset(df_train, hdf5_path,
                               img_size=input_size,
                               mask_size=input_size)

    test_dataset = GanDataset(df_test, hdf5_path,
                              img_size=input_size,
                              mask_size=input_size)

    batch_size = 16
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
    num_epochs = 500
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=0.01,
                                  amsgrad=False)

    criterion = torch.nn.SmoothL1Loss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=num_epochs,
                                                           eta_min=learning_rate/10)

    desired_metrics = ['loss']
    splits = ['train', 'test']
    history = {'epoch': []}
    for metric_name in desired_metrics:
        for split in splits:
            history[f'{split}_{metric_name}'] = []

    img_gif = []
    with tqdm(total=num_epochs, desc='epoch', position=0) as pbar:
        for epoch in range(0, num_epochs):  
            total_train_loss = 0
            model.train()
            optimizer.zero_grad()

            for images, targets, masks in tqdm(train_loader, position=1, leave=False, desc='train batch'):
                targets = targets.to(device)
                recon, latent = model(images.to(device))
                recon = F.sigmoid(recon)
                #loss = criterion(recon, targets)
                loss = criterion(recon[masks>0], targets[masks>0])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            scheduler.step()

            # test loop
            total_test_loss = 0
            model.eval()
            with torch.no_grad():
                for images, targets, masks in tqdm(test_loader, position=1, leave=False, desc='test batch'):
                    targets = targets.to(device)
                    recon, latent = model(images.to(device))
                    recon = F.sigmoid(recon)
                    #loss = criterion(recon, targets)
                    loss = criterion(recon[masks>0], targets[masks>0])
                    total_test_loss += loss.item()
            
            mask_recon = masks[0, 0, :, :].detach().cpu().numpy() 
            im_recon = recon[0, 0, :, :].detach().cpu().numpy()
            im_recon = (np.clip(im_recon, 0, 1)*255).astype(np.uint8)
            
            im_merged = images[0, 0, :, :].detach().cpu().numpy()
            im_merged = (np.clip(im_merged, 0, 1)*255).astype(np.uint8)
            
            im_merged[mask_recon>0] = im_recon[mask_recon>0]
         
            img_gif.append(im_merged)

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
                torch.save(model.state_dict(), os.path.join('models', 'best_model.pth'))
                df_history.to_csv(os.path.join('models', 'training_history.csv'))
            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            #if epoch % 10 == 0:
            if epoch > -1:
                io.imshow(im_recon)
                io.show()

                io.imshow(im_merged)
                io.show()

                df_history[['train_loss', 'test_loss']].plot()
                io.show()

    df_history[['train_loss', 'test_loss']].plot()
    io.show()

    imageio.mimsave('img_gif.gif', img_gif, loop=0)
