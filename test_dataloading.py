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

train = mods.H5SpecDataset(DATA)
valid = mods.H5SpecDataset(DATA, split = "validation")
test = mods.H5SpecDataset(DATA, split = "test")

train_loader = torch.utils.data.DataLoader(train, batch_size = 2, shuffle = True, num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid, batch_size = 1, shuffle = False,)
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False,)
#################################################################################################

###############
###############
TESTING = False
###############
###############

# intiate test paramters and stuff

INPUT_SIZE = train[0][0].shape[0]
CONFIG = [
    {'in': 256,   'out': 64, },
]
LATENT_SIZE = 32
ACTIVATION_FUNCTION = 'ReLU'
EPOCHS = 50
EARLY_STOPPING = False
BETA = 0 # kl weighting only used in VAE, automatically set as 0 for other models
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8

#####################################
############# make model ############
#####################################
model = mods.StandardAutoencoder(CONFIG, INPUT_SIZE, LATENT_SIZE, activation = ACTIVATION_FUNCTION)
#####################################

TEST_NAME = f'RUN_{model.type}_nl{len(CONFIG)}_ls{LATENT_SIZE}_e{EPOCHS}_{ACTIVATION_FUNCTION}_B{BETA}_lr{LEARNING_RATE:.0e}_wd{WEIGHT_DECAY}_es{EARLY_STOPPING}'

test_params = {
    'test name': TEST_NAME,
    'data file': DATA,
    'ae type' : model.type,
    'config' : CONFIG,
    'latent size' : LATENT_SIZE,
    'activation function' : ACTIVATION_FUNCTION,
    'max epochs' : EPOCHS,
    'beta' : BETA,
    'learn rate' : LEARNING_RATE,
    'weight decay' : WEIGHT_DECAY
}

funcs.save_test_params(test_params, TEST_NAME, test=TESTING)

######################################################################################################

print(model)

if EARLY_STOPPING:
    early_stopping = mods.CustomEarlyStopping(TEST_NAME, patience = 10, delta = 0.0, test = TESTING, verbose = True)
else:
    early_stopping = None

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# train
torch.cuda.empty_cache()
model, model_losses = funcs.train_ae(EPOCHS, train_loader, valid_loader, model, optimizer, early_stopping = early_stopping, beta=BETA, verbose = True, )


funcs.plot_loss(model_losses, test_params['test name'], test=TESTING)

funcs.plot_examples(train_loader, model, test_params, test = TESTING)
