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

import funcs
from funcs import train_ae
import mods
from mods import SpecDataset
from mods import LoadData


def generate_sine_data(num_samples, seq_len):
    x = np.linspace(0, 2 * np.pi, seq_len, dtype =  np.float32)
    data = []
    for _ in range(num_samples):
        # Random phase and slight amplitude shift
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.8, 1.2)
        sample = amp * np.sin(x + phase)
        data.append(sample)
    return np.array(data, dtype =  np.float32), x

# Creates 1000 samples of 1D vectors with 64 points each
test_data, l = generate_sine_data(1000, 64)

f_train, f_test = train_test_split(test_data)
# print(type(f_train))

# # z score standardise
# MU = float(f_train.mean())
# SIGMA = float(f_train.std())
# f_train = (f_train - MU) / SIGMA
# f_test = (f_test - MU) / SIGMA

# min-max (normalization)
f_min = np.min(f_train)
f_max = np.max(f_train)
f_train = (f_train - f_min)/(f_max - f_min)
f_test = (f_test - f_min)/(f_max - f_min)

f_train = np.asarray(f_train)
f_test = np.asarray(f_test)

f_train = torch.from_numpy(f_train)
f_test = torch.from_numpy(f_test)

train_fluxes = f_train
test_fluxes = f_test

train_dataset = SpecDataset(train_fluxes)
test_dataset = SpecDataset(test_fluxes)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# print(train_loader.dataset.data)


INPUT_SIZE = len(train_fluxes[1])

test_config = [
    {'in': 5000,   'out': 2000, },
    {'in': 2000,  'out': 1000, },
    {'in': 1000,  'out': 500, },
    {'in': 500,  'out' : 256, },
]

LATENT_SIZE = 128

model = mods.VAEAutoencoder(test_config, INPUT_SIZE, LATENT_SIZE)
# model = mods.StandardAutoencoder(test_config, INPUT_SIZE, LATENT_SIZE)

# model.to(device)
print(model)

# train

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

EPOCHS = 5
beta = 0 # kl weighting

model, model_losses = train_ae(EPOCHS, train_loader, test_loader, model, optimizer, beta=beta, verbose = True, )


# funcs.plot_loss(model_losses)

# funcs.plot_examples(train_loader, model, l, 'minmax', f_min, f_max)