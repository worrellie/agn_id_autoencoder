

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

# then get TEST DATA results for analysis
# reconstruction loss, min, max, mean, median
# reconstruction loss histogram
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = 
normalize = 

# load test data
test = mods.H5SpecDataset(DATA, split = "test")
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False,)

# load model

model.to(device)

torch.load(model)
recon_losses = []

self.model.eval()
with torch.no_grad():

    for x, x_mask in train_loader:

        if normalize:
            x = (x - train_mean) / train_std # normalize data
            x = x * x_mask # re-set 'gaps'/masked regions as zero

        x = x.to(self.device)
        x_mask = x_mask.to(self.device)

        x_hat, mu, logvar = self.model(x) # batch prediction

        reconstruction_loss, _, _ = _loss_calc_spec(x_hat, x, x_mask, )
