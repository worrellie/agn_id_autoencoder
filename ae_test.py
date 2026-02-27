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




scaler = StandardScaler()
loader_of_data = mods.LoadData(scaler, spec_dir ="/home/worrellie/Documents/phd/autoencoder/test_gal")

raw_train, raw_valid, raw_test = loader_of_data.load_raw()
train, valid, test = loader_of_data.scale_raw(raw_train, raw_valid, raw_test)

l = loader_of_data.l

train_dataset = SpecDataset(train)
valid_dataset = SpecDataset(valid)
test_dataset = SpecDataset(test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

TEST_NAME = "standard_test_gal_relu"

SCALING = 'zscore'

INPUT_SIZE = len(train[1])

CONFIG = [
    {'in': 5000,   'out': 2000, },
    {'in': 2000,  'out': 1000, },
]

LATENT_SIZE = 500
ACTIVATION_FUNCTION = 'ReLU'

EPOCHS = 10

BETA = 1e-4 # kl weighting

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-8

test_params = {
    'test_name': TEST_NAME,
    'scaling' : SCALING,
    'config' : CONFIG,
    'latent_size' : LATENT_SIZE,
    'activation_function' : ACTIVATION_FUNCTION,
    'epochs' : EPOCHS,
    'beta' : BETA,
    'lr' : LEARNING_RATE,
    'weight_decay' : WEIGHT_DECAY
}


testing = False

funcs.save_test_params(test_params, TEST_NAME, test=testing)

###############################################

model = mods.StandardAutoencoder(CONFIG, INPUT_SIZE, LATENT_SIZE, activation = ACTIVATION_FUNCTION)
print(model)

# early_stopping = mods.CustomEarlyStopping(TEST_NAME, patience = 5, delta = 0, test = testing, verbose = True)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

###############################################

# train

model, model_losses = train_ae(EPOCHS, train_loader, test_loader, model, optimizer, beta=BETA, verbose = True, )

funcs.plot_loss(model_losses, test_params['test_name'], test=testing)

MU = loader_of_data.scaler
SIGMA = loader_of_data.scaler

funcs.plot_examples(train_loader, model, l, test_params, MU, SIGMA, test = testing)

