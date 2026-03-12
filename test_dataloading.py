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

import funcs
import mods

# Load data

DATA = "all_spectra_normalized.h5"

# with h5py.File(DATA, 'r') as hf:
    # l = hf.attrs['wavelengths'][:]
    # train_mean = hf.attrs['train_mean']
    # train_std = hf.attrs['train_std']

print('making datasets')

train = mods.H5SpecDataset(DATA)
valid = mods.H5SpecDataset(DATA, split = "validation")
test = mods.H5SpecDataset(DATA, split = "test")

print('making loaders')

train_loader = torch.utils.data.DataLoader(train, batch_size = 2, shuffle = True, num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid, batch_size = 1, shuffle = False,)
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False,)


# print('getting global stats')
# train_mean, train_std = funcs.global_stats(train_loader)
#################################################################################################

print('saving test params')

# intiate test paramters and stuff
TEST_NAME = "test"

INPUT_SIZE = train[0][0].shape[0]
CONFIG = [
    {'in': 256,   'out': 64, },
]
LATENT_SIZE = 32
ACTIVATION_FUNCTION = 'ReLU'
EPOCHS = 5
EARLY_STOPPING = False
BETA = 1e-4 # kl weighting only used in VAE, set as 0 for other models
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8

test_params = {
    'test_name': TEST_NAME,
    'data file': DATA,
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

######################################################################################################

model = mods.StandardAutoencoder(CONFIG, INPUT_SIZE, LATENT_SIZE, activation = ACTIVATION_FUNCTION)
print(model)

if EARLY_STOPPING:
    early_stopping = mods.CustomEarlyStopping(TEST_NAME, patience = 10, delta = 0.0, test = TESTING, verbose = True)
else:
    early_stopping = None

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# train
torch.cuda.empty_cache()
model, model_losses = funcs.train_ae(EPOCHS, train_loader, valid_loader, model, optimizer, early_stopping = early_stopping, beta=BETA, verbose = True, )


funcs.plot_loss(model_losses, test_params['test_name'], test=TESTING)

funcs.plot_examples(train_loader, model, test_params, test = TESTING)
# funcs.plot_examples(valid_loader, model, l, test_params, SCALING, test = TESTING)
