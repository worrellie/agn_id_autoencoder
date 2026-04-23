import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import pathlib as path
import h5py

####### autoencoders ########

class StandardAutoencoder(nn.Module):

    def __init__(self, config, input_size, latent_size, activation = 'ReLU'):
        super(StandardAutoencoder, self).__init__()

        self.act_func = getattr(nn, activation)() # make instance of desired activation function

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.type = 'sae'

        self.input_to_encoder = nn.Linear(input_size, config[0]['in'])

        # add encoder layers
        for c in config:
            self.encoder_layers.append(
                nn.Linear(c['in'], c['out'], )
            )

        self.encoder_to_latent= nn.Linear(config[-1]['out'], latent_size)


        # add decoder layers
        for c in reversed(config):
            self.decoder_layers.append(
                nn.Linear(c['out'], c['in'], )
            )
        
        self.decoder_from_latent = nn.Linear(latent_size, config[-1]['out'])

        self.decoder_to_output = nn.Linear(config[0]['in'], input_size)

    def forward(self, x):

        # print(x.shape)
        # x = torch.relu(self.input_to_encoder(x))
        x = self.act_func(self.input_to_encoder(x))

        # print(x.shape)

        for l in self.encoder_layers:
            # x = torch.relu(l(x))
            x = self.act_func(l(x))
            # print(x.shape)

        # z = torch.relu(self.encoder_to_latent(x))
        z = self.act_func(self.encoder_to_latent(x))
        # print(z.shape)

        # z = torch.relu(self.decoder_from_latent(z))
        z = self.act_func(self.decoder_from_latent(z))
        # print(z.shape)


        for l in self.decoder_layers:
            # z = torch.relu(l(z))
            z = self.act_func(l(z))
            # print(z.shape)

        x_hat = self.decoder_to_output(z)
        # print(x_hat.shape)

        return x_hat, None, None


class VAEAutoencoder(nn.Module):

    def __init__(self, config, input_size, latent_size, activation = 'ReLU'):
        super(VAEAutoencoder, self).__init__()

        self.type = 'vae'

        self.act_func = getattr(nn, activation)() # make instance of desired activation function

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.input_to_encoder = nn.Linear(input_size, config[0]['in'])

        # add encoder layers
        for c in config:
            self.encoder_layers.append(
                nn.Linear(c['in'], c['out'], )
            )

    # ###### this is cool- remember for future
    # def _get_flattened_size(self, input_size):
        
    #     with torch.no_grad(): # do not update weights
            
    #         dummy_x = torch.zeros(1, 1, input_size)
    #         for l in self.encoder_layers:
    #             dummy_x = l(dummy_x) # updates the dummy shape based on the encoder layers
            
    #         return dummy_x.numel(), dummy_x.shape[1], dummy_x.shape[2]
    # ######
        
        
        # add decoder layers
        for c in reversed(config):
            self.decoder_layers.append(
                nn.Linear(c['out'], c['in'], )
            )
        
        # add latent layers
        self.encoder_to_latent_mean = nn.Linear(config[-1]['out'], latent_size)
        self.encoder_to_latent_logvar = nn.Linear(config[-1]['out'], latent_size)
        
        self.decoder_from_latent = nn.Linear(latent_size, config[-1]['out'])

        self.decoder_to_output = nn.Linear(config[0]['in'], input_size)


    def forward(self, x):
        
        # x = torch.relu(self.input_to_encoder(x))
        x = self.act_func(self.input_to_encoder(x))

        for l in self.encoder_layers:
            # x = torch.relu(l(x))
            x = self.act_func(l(x))

        # mu = torch.relu(self.encoder_to_latent_mean(x))
        mu = self.encoder_to_latent_mean(x)
        # logvar = torch.relu(self.encoder_to_latent_logvar(x))
        logvar = self.encoder_to_latent_logvar(x)
        
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std*epsilon # latent of VAE

        # z = torch.relu(self.decoder_from_latent(z))
        z = self.act_func(self.decoder_from_latent(z))

        for l in self.decoder_layers:
            # z = torch.relu(l(z))
            z = self.act_func(l(z))

        x_hat = self.decoder_to_output(z)

        return x_hat, mu, logvar


# class CNNAutoencoder(nn.Module):

#     def __init__(self, config, input_size, latent_size):
#         super(CNNAutoencoder, self).__init__()

#         self.encoder_layers = nn.ModuleList()
#         self.decoder_layers = nn.ModuleList()

#         # add encoder layers
#         for c in config:
#             self.encoder_layers.append(
#                 nn.Conv1d(c['in'], c['out'], c['kernel'], 
#                           stride=c['stride'], padding=c['padding'])
#             )

#         self.flat_size, self.last_chans, self.last_width = self._get_flattened_size(input_size)

#         # add latent layers
#         self.encoder_to_latent_mean = nn.Linear(self.flat_size, latent_size)
#         self.encoder_to_latent_logvar = nn.Linear(self.flat_size, latent_size)
#         self.decoder_from_latent = nn.Linear(latent_size, self.flat_size)

#         # add decoder layers
#         for c in reversed(config):
#             self.decoder_layers.append(
#                 nn.ConvTranspose1d(c['out'], c['in'], c['kernel'], 
#                           stride=c['stride'], padding=c['padding'])
#             )

#     ###### this is cool- remember for future
#     def _get_flattened_size(self, input_size):
        
#         with torch.no_grad(): # do not update weights
            
#             dummy_x = torch.zeros(1, 1, input_size)
#             for l in self.encoder_layers:
#                 dummy_x = l(dummy_x) # updates the dummy shape based on the encoder layers
            
#             return dummy_x.numel(), dummy_x.shape[1], dummy_x.shape[2]
#     ######

#     def forward(self, x):

#         x_shapes_encoder = []
#         x_chans_encoder = []
#         x = x.unsqueeze(1) # add channel dimension
#         for l in self.encoder_layers:
#             x = torch.relu(l(x))
#             # print(x.shape)
#             x_shapes_encoder.append(x.shape[-1])
#             x_chans_encoder.append(x.shape[-2])

#         x = torch.flatten(x, start_dim = 1)
#         # flat = x.shape[1]
#         # print((f'flat: {flat}'))
#         # print(x.shape)

#         mu = torch.relu(self.encoder_to_latent_mean(x))
#         logvar = torch.relu(self.encoder_to_latent_logvar(x))
#         # print(mu.shape)
#         # print(logvar.shape)

#         # epsilon = torch.randn_like(logvar).to(device)
#         epsilon = torch.randn_like(logvar)
#         z = mu + logvar*epsilon # latent of VAE
#         # print(z.shape)

#         # print(x_shapes_encoder)

#         # self.decoder_from_latent = nn.LazyLinear(flat).to(device)
#         # self.decoder_from_latent = nn.LazyLinear(flat)
#         z = torch.relu(self.decoder_from_latent(z))
#         # print(z.shape)

#         z = z.view(-1, x_chans_encoder[-1], x_shapes_encoder[-1]) # reshape for decoder
#         # print(z.shape)


#         for l in self.decoder_layers:
#             z = torch.relu(l(z))
#             # print(z.shape)

#         x_hat = z.squeeze(1) # remove (now defunct) channel dimension

#         return x_hat, mu, logvar