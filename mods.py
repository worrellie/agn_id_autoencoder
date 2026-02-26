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

import funcs
import mods

class SpecDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data[idx]

        return sample, sample

####### autoencoders ########

class StandardAutoencoder(nn.Module):

    def __init__(self, config, input_size, latent_size, activation = 'ReLU'):
        super(StandardAutoencoder, self).__init__()

        self.act_func = getattr(nn, activation)() # make instance of desired activation function

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

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

        # x = torch.relu(self.input_to_encoder(x))
        x = self.act_func(self.input_to_encoder(x))

        for l in self.encoder_layers:
            # x = torch.relu(l(x))
            x = self.act_func(l(x))

        # z = torch.relu(self.encoder_to_latent(x))
        z = self.act_func(self.encoder_to_latent(x))

        # z = torch.relu(self.decoder_from_latent(z))
        z = self.act_func(self.decoder_from_latent(z))


        for l in self.decoder_layers:
            # z = torch.relu(l(z))
            z = self.act_func(l(z))

        x_hat = self.decoder_to_output(z)

        return x_hat, None, None


class VAEAutoencoder(nn.Module):

    def __init__(self, config, input_size, latent_size, activation = 'ReLU'):
        super(VAEAutoencoder, self).__init__()

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


################## autoencoder #####################

class CNNAutoencoder(nn.Module):

    def __init__(self, config, input_size, latent_size):
        super(CNNAutoencoder, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # add encoder layers
        for c in config:
            self.encoder_layers.append(
                nn.Conv1d(c['in'], c['out'], c['kernel'], 
                          stride=c['stride'], padding=c['padding'])
            )

        self.flat_size, self.last_chans, self.last_width = self._get_flattened_size(input_size)

        # add latent layers
        self.encoder_to_latent_mean = nn.Linear(self.flat_size, latent_size)
        self.encoder_to_latent_logvar = nn.Linear(self.flat_size, latent_size)
        self.decoder_from_latent = nn.Linear(latent_size, self.flat_size)

        # add decoder layers
        for c in reversed(config):
            self.decoder_layers.append(
                nn.ConvTranspose1d(c['out'], c['in'], c['kernel'], 
                          stride=c['stride'], padding=c['padding'])
            )

    ###### this is cool- remember for future
    def _get_flattened_size(self, input_size):
        
        with torch.no_grad(): # do not update weights
            
            dummy_x = torch.zeros(1, 1, input_size)
            for l in self.encoder_layers:
                dummy_x = l(dummy_x) # updates the dummy shape based on the encoder layers
            
            return dummy_x.numel(), dummy_x.shape[1], dummy_x.shape[2]
    ######

    def forward(self, x):

        x_shapes_encoder = []
        x_chans_encoder = []
        x = x.unsqueeze(1) # add channel dimension
        for l in self.encoder_layers:
            x = torch.relu(l(x))
            # print(x.shape)
            x_shapes_encoder.append(x.shape[-1])
            x_chans_encoder.append(x.shape[-2])

        x = torch.flatten(x, start_dim = 1)
        # flat = x.shape[1]
        # print((f'flat: {flat}'))
        # print(x.shape)

        mu = torch.relu(self.encoder_to_latent_mean(x))
        logvar = torch.relu(self.encoder_to_latent_logvar(x))
        # print(mu.shape)
        # print(logvar.shape)

        # epsilon = torch.randn_like(logvar).to(device)
        epsilon = torch.randn_like(logvar)
        z = mu + logvar*epsilon # latent of VAE
        # print(z.shape)

        # print(x_shapes_encoder)

        # self.decoder_from_latent = nn.LazyLinear(flat).to(device)
        # self.decoder_from_latent = nn.LazyLinear(flat)
        z = torch.relu(self.decoder_from_latent(z))
        # print(z.shape)

        z = z.view(-1, x_chans_encoder[-1], x_shapes_encoder[-1]) # reshape for decoder
        # print(z.shape)


        for l in self.decoder_layers:
            z = torch.relu(l(z))
            # print(z.shape)

        x_hat = z.squeeze(1) # remove (now defunct) channel dimension

        return x_hat, mu, logvar

class LoadData():

    def __init__(self, std, spec_dir ="/home/worrellie/Documents/phd/autoencoder/merged_spectra_gal",
                 agn_dir="/home/worrellie/Documents/phd/autoencoder/agn/merged_spectra_agn"):
        self.std = std
        self.spec_dir = spec_dir
        self.agn_dir = agn_dir

    def load_galaxies(self,):

        fluxes = []
        i=0
        for spec in os.listdir(self.spec_dir):
            i= i+1
            spec_path = os.path.join(self.spec_dir, spec)
            try:
                with fits.open(spec_path) as hdul:

                    data = hdul[1].data
                    flux = data['flux']
                    # print(flux)
                    l = data['lambda']
                    self.l = l
                    flux = flux.astype(np.float32)
                    flux = torch.from_numpy(flux)
                    fluxes.append(flux)
                    # if i == 1:
                    #     plt.figure()
                    #     plt.plot(l,flux)
                    #     # plt.show()
        

            except Exception as e:
                print(f"Error opening spectrum: {spec} ({e})")
            

        fluxes = np.asarray(fluxes)
        INPUT_SIZE = len(fluxes[1])

        f_train, f_test = train_test_split(fluxes)
        f_train, f_valid = train_test_split(f_train, test_size = 0.1)

        # plt.figure()
        # plt.plot(l, f_train[1])
        # plt.title('pre-standardized example')
        # plt.show()

        # standardize
        if self.std == 'zscore':

            self.MU = float(f_train.mean())   # MU and SIGMA of training only
            self.SIGMA = float(f_train.std()) # otherwise have data leakage

            f_train = (f_train - self.MU) / self.SIGMA # standardized fluxes of TRAINING ONLY
            f_test = (f_test - self.MU) / self.SIGMA
            f_valid = (f_valid - self.MU) / self.SIGMA

        elif self.std == 'minmax': # normalization

            self.f_min = min(f_train)
            self.f_max = max(f_train)

            f_train = (f_train - self.f_min)/ (self.f_max - self.f_min)
            f_test = (f_test - self.f_min)/ (self.f_max - self.f_min)
            f_valid = (f_valid - self.f_min)/ (self.f_max - self.f_min)


        # plt.figure()
        # plt.plot(l, f_train[1])
        # plt.title('standardized example')
        # plt.show()
        # exit()

        f_train = np.asarray(f_train)
        f_test = np.asarray(f_test)
        f_valid = np.asarray(f_valid)

        f_train = torch.from_numpy(f_train)
        f_valid = torch.from_numpy(f_valid)
        f_test = torch.from_numpy(f_test)


        return f_train, f_valid, f_test


    def load_agn(self,):

        fluxes_agn = []

        for spec in os.listdir(self.agn_dir):
            spec_path = os.path.join(self.agn_dir, spec)
            if "z0.9" in spec_path:
                try:
                    with fits.open(spec_path) as hdul:
                        data = hdul[1].data
                        flux = data['flux']
                        flux = flux.astype(np.float32)
                        flux = torch.from_numpy(flux)
                        fluxes_agn.append(flux)

                except Exception as e:
                    print(f"Error opening spectrum: {spec} ({e})")

        fluxes_agn = np.asarray(fluxes_agn)

        fluxes_agn_std = (fluxes_agn - self.MU) / self.SIGMA # standardized fluxes

        f_agn = np.asarray(fluxes_agn_std)

        f_agn = torch.from_numpy(f_agn)

        return f_agn

class CustomEarlyStopping:

    def __init__(self, test_name, patience = 5, delta = 0, test = False, verbose = False):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improve_count = 0
        self.stop_training = False
        self.test = test
        self.test_name = test_name

    def save_model(self, model, test_name, epoch):
        
        save_path = path.Path(test_name, f"{test_name}_{self.patience}_{self.delta}.pt") # overwrite is default

        torch.save(model.state_dict(), save_path)
    
    def check_early_stop(self, validation_loss, model, epoch):

        if self.best_loss is None or validation_loss < self.best_loss - self.delta:
            # if starting or if new validation loss is better than current best loss
            # set best loss as new validaton loss and reset count of no improvement
            # and save best model
            self.best_loss = validation_loss
            if not self.test:
                self.save_model(model, self.test_name, epoch)
            self.no_improve_count = 0
        else:
            # if new validation loss is not better, increase count of no improvement
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                # no improve count reaches patience, stop
                self.stop_training = True
                if self.verbose:
                    print('Stopping Early')
