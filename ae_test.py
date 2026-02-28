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

scaler = StandardScaler()
loader_of_data = mods.LoadData(scaler, spec_dir ="/home/worrellie/Documents/phd/autoencoder/merged_z09_v3-001")

raw_train, raw_valid, raw_test = loader_of_data.load_raw()
train, valid, test = loader_of_data.scale_raw(raw_train, raw_valid, raw_test)

l = loader_of_data.l

train_dataset = SpecDataset(train)
valid_dataset = SpecDataset(valid)
test_dataset = SpecDataset(test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

TEST_NAME = "all_z09_relu_no_early_stop"

SCALING = loader_of_data.scaler

INPUT_SIZE = len(train[1])

CONFIG = [
    {'in': 5000,   'out': 3000, },
    {'in': 3000,  'out': 2000, },
]

LATENT_SIZE = 500
ACTIVATION_FUNCTION = 'ReLU'

EPOCHS = 500

BETA = 1e-4 # kl weighting only used in VAE

LEARNING_RATE = 1e-4
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
# print(funcs.get_model_size_mb(model))
# model.half()
# print(funcs.get_model_size_mb(model))
# exit()

# early_stopping = mods.CustomEarlyStopping(TEST_NAME, patience = 10, delta = 0.0, test = testing, verbose = True)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

###############################################

# train
torch.cuda.empty_cache()
model, model_losses = train_ae(EPOCHS, train_loader, test_loader, model, optimizer, beta=BETA, verbose = True, )

funcs.plot_loss(model_losses, test_params['test_name'], test=testing)

MU = loader_of_data.scaler
SIGMA = loader_of_data.scaler

funcs.plot_examples(train_loader, model, l, test_params, SCALING, test = testing)

