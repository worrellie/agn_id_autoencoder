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
from sklearn.preprocessing import StandardScaler

import funcs
from funcs import train_ae
import mods
from mods import SpecDataset
from mods import LoadData

from torch.cuda.amp import autocast

# DIR = "/home/worrellie/Documents/phd/autoencoder/merged_z09_v3-001_linearly_scaled_noiseless_1h"
# DIR = "/home/worrellie/Documents/phd/autoencoder/merged_z09_v3-001_linearly_scaled_noiseless_8h"
DIR = "/home/worrellie/Documents/phd/autoencoder/merged_z09_v3-001_linearly_scaled_noisy_1h"
# DIR = "/home/worrellie/Documents/phd/autoencoder/merged_z09_v3-001_linearly_scaled_noisy_8h"

scaler = StandardScaler() # initiate scaler for data
loader_of_data = mods.LoadData(scaler, spec_dir =DIR) # initiate data loader

raw_train, raw_valid, raw_test = loader_of_data.load_raw() # load raw fluxes (returned as np arrays)
train, valid, test = loader_of_data.scale_raw(raw_train, raw_valid, raw_test) # scale according to scaler used

l = loader_of_data.l # get wavelength array
print(min(l))
print(max(l))
exit()

train_dataset = SpecDataset(train) # put scaled fluxes into Pytorch Dataset
valid_dataset = SpecDataset(valid)
test_dataset = SpecDataset(test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) # dataset into Pytorch dataloader
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# intiate test paramters and stuff

SCALING = loader_of_data.scaler
INPUT_SIZE = len(train[1])
CONFIG = [
    {'in': 256,   'out': 64, },
]
LATENT_SIZE = 32
ACTIVATION_FUNCTION = 'ReLU'
EPOCHS = 50
EARLY_STOPPING = False
BETA = 1e-4 # kl weighting only used in VAE, set as 0 for other models
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8

TEST_NAME = "report_specifications"
# TEST_NAME = f'z09_1h_{ACTIVATION_FUNCTION}_{EPOCHS}e_es{EARLY_STOPPING}_ls{LATENT_SIZE}'

test_params = {
    'test_name': TEST_NAME,
    'data_dir': DIR,
    'scaling' : SCALING,
    'config' : CONFIG,
    'latent_size' : LATENT_SIZE,
    'activation_function' : ACTIVATION_FUNCTION,
    'epochs' : EPOCHS,
    'beta' : BETA,
    'learn_rate' : LEARNING_RATE,
    'weight_decay' : WEIGHT_DECAY
}

TESTING = False

funcs.save_test_params(test_params, TEST_NAME, test=TESTING)

###############################################

model = mods.StandardAutoencoder(CONFIG, INPUT_SIZE, LATENT_SIZE, activation = ACTIVATION_FUNCTION)
print(model)
# print(funcs.get_model_size_mb(model))
# model.half()
# print(funcs.get_model_size_mb(model))
# exit()

if EARLY_STOPPING:
    early_stopping = mods.CustomEarlyStopping(TEST_NAME, patience = 10, delta = 0.0, test = testing, verbose = True)
else:
    early_stopping = None

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

###############################################

# train
torch.cuda.empty_cache()
model, model_losses = train_ae(EPOCHS, train_loader, test_loader, model, optimizer, early_stopping = early_stopping, beta=BETA, verbose = True, )

funcs.plot_loss(model_losses, test_params['test_name'], test=TESTING)

MU = loader_of_data.scaler
SIGMA = loader_of_data.scaler

funcs.plot_examples(train_loader, model, l, test_params, SCALING, test = TESTING)
# funcs.plot_examples(valid_loader, model, l, test_params, SCALING, test = TESTING)

