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

train = mods.H5SpecDataset("test_all_spectra_normalized.h5")
valid = mods.H5SpecDataset("test_all_spectra_normalized.h5", split = "validation")
test = mods.H5SpecDataset("test_all_spectra_normalized.h5", split = "test")

# print(train.__getitem__(1)[1])

train_loader = torch.utils.data.DataLoader(train, batch_size = 1, shuffle = True, num_workers = 0)
# get mean and std to normalize data (in training loop)
train_mean, train_std = funcs.global_stats(train_loader)
# print(train_mean, train_std)

valid_loader = torch.utils.data.DataLoader(valid, batch_size = 1, shuffle = False,)
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False,)

