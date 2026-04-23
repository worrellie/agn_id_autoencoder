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
from datahandling import H5SpecDataset
import autoencoder as ae
import training

import time

# Load data

DATA = "test_all_spectra.h5"

# default is normalised data
train = H5SpecDataset(DATA, split = "train")
valid = H5SpecDataset(DATA, split = "validation")

# # 
# for nw in [0, 2, 4, 8, 12]:
#     train_loader = torch.utils.data.DataLoader(train, batch_size=64, num_workers=nw)
#     start = time.time()
    
#     for i, data in enumerate(train_loader):
#         if i > 10: break  # Just test the first few batches
        
#     end = time.time()
#     print(f"num_workers: {nw} | Time per 10 batches: {end - start:.4f}s")
# # 

train_loader = torch.utils.data.DataLoader(train, batch_size = 2, shuffle = True, num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid, batch_size = 1, shuffle = False,)

#################################################################################################

###############
###############
TESTING = True
verb = TESTING

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
BETA = 1e4 # kl weighting only used in VAE, automatically set as 0 for other models
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8

#####################################
############# make model ############
#####################################
model = ae.VAEAutoencoder(CONFIG, INPUT_SIZE, LATENT_SIZE, activation = ACTIVATION_FUNCTION)
#####################################

TEST_NAME = f'RUN_{model.type}_nl{len(CONFIG)}_ls{LATENT_SIZE}_e{EPOCHS}_{ACTIVATION_FUNCTION}_B{BETA:.0e}_lr{LEARNING_RATE:.0e}_wd{WEIGHT_DECAY}_es{EARLY_STOPPING}'

test_params = {
    'test_name': TEST_NAME,
    'data_file': DATA,
    'ae_type' : model.type,
    'config' : CONFIG,
    'latent_size' : LATENT_SIZE,
    'activation_function' : ACTIVATION_FUNCTION,
    'max_epochs' : EPOCHS,
    'beta' : BETA,
    'learn_rate' : LEARNING_RATE,
    'weight_decay' : WEIGHT_DECAY
}

funcs.save_test_params(test_params, TEST_NAME, test=TESTING)

######################################################################################################

print(model)

if EARLY_STOPPING:
    early_stopping = training.CustomEarlyStopping(TEST_NAME, patience = 10, delta = 0.0, test = TESTING, verbose = verb)
else:
    early_stopping = None

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# train
torch.cuda.empty_cache()
# model, model_losses = funcs.train_ae(EPOCHS, train_loader, valid_loader, model, optimizer, early_stopping = early_stopping, beta=BETA, verbose = verb, )
start = time.time()
trainer = training.Trainer(device, TEST_NAME, model, optimizer, early_stopping, BETA, test = TESTING)
model, model_losses = trainer.train_ae(EPOCHS, train_loader, valid_loader = valid_loader, verbose = verb)
stop = time.time()

print(f"{stop-start} seconds to train")

funcs.plot_loss(model_losses, test_params['test_name'], test=TESTING)

funcs.plot_examples(train_loader, model, test_params, test = TESTING)
