# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:57:34 2024

@author: Mico
"""

import torch
from torch import nn
from torch.nn import functional as F

def get_fmap_size(in_size, kernel, padding, stride):
    out_size = 1 + ((in_size - kernel + 2 *padding) / stride)
    return out_size


class L1Regularization(nn.Module):
    def __init__(self, sparsity=1e-5):
        super(L1Regularization, self).__init__()
        self.sparsity = sparsity

    def forward(self, latent):
        loss = self.sparsity * torch.mean(torch.abs(latent))
        return loss

class AutoEncoder(nn.Module):
    def __init__(self, input_size=512, channels=[64, 128, 256], use_transpose_conv=True):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, [1] + channels)
        self.decoder = Decoder(input_size, channels[::-1] + [16], use_transpose_conv)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    
class TConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TConvBn, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, channels):
        super(Encoder, self).__init__()
        conv_layers = []
        for i in range(0, len(channels)-1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            conv_layers.append(ConvBn(in_channels=in_ch, out_channels=out_ch,
                                         kernel_size=4, stride=2, padding=1))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, input_size, channels, use_transpose_conv=True):
        super(Decoder, self).__init__()
        self.fmap_size = (input_size // 2**(len(channels)-1))  
        in_features_size = self.fmap_size * self.fmap_size * channels[0]

        deconv_layers = []
        for i in range(0, len(channels)-1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            if use_transpose_conv:
                deconv_layers.append(TConvBn(in_channels=in_ch, out_channels=out_ch,
                                             kernel_size=4, stride=2, padding=1))
            else:
                deconv_layers.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                                   nn.Conv2d(in_channels=in_ch,
                                                             out_channels=out_ch,
                                                             kernel_size=3, stride=1, padding=1),
                                                   nn.BatchNorm2d(out_ch)))
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.head = nn.Conv2d(in_channels=channels[-1], out_channels=1, kernel_size=1)



    def forward(self, x):
        for deconv_layer in self.deconv_layers:
            x = F.relu(deconv_layer(x))
        x = self.head(x)
        return x
