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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data

data_loader = LoadData()

train_fluxes, valid_fluxes, test_fluxes = data_loader.load_galaxies()

train_dataset = SpecDataset(train_fluxes)
valid_dataset = SpecDataset(valid_fluxes)
test_dataset = SpecDataset(test_fluxes)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# make model

LATENT_SIZE = 10
INPUT_SIZE = len(train_fluxes[1])

test_config = [
    {'in': 5000,   'out': 2000, },
    {'in': 2000,  'out': 1000, },
    {'in': 1000,  'out': 500, },
    {'in': 500,  'out': 128, },
    {'in': 128,  'out': 64, },
    {'in': 64,  'out': 32, },
]

model = mods.VAEAutoencoder(test_config, INPUT_SIZE, LATENT_SIZE)
# model.to(device)
print(model)

# train

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

EPOCHS = 5

model, train_losses, valid_losses = train_ae(EPOCHS, train_loader, valid_loader, model, optimizer,  verbose = True, )

# funcs.plot_loss(train_losses, valid_losses)

# example_idxs = funcs.get_example_specs(train_losses)

# predictions
model.to(device)
model.eval()
output = {'recon':[],'og':[], 'loss':[]}
for x, _ in train_dataset:
    
    # x = x.tolist()
    # plt.figure()
    # plt.plot(data_loader.l, x)
    # plt.title('quoi??')
    # plt.show()
    # x = torch.tensor(x)

    x = x.to(device)
    x_hat, mu, logvar = model(x)

    # x_hat= x_hat.detach().cpu().tolist()
    # plt.figure()
    # plt.plot(data_loader.l, x_hat)
    # plt.show()
    # x_hat = torch.tensor(x_hat)

    loss = funcs.loss_calc(x_hat, x, mu, logvar).detach().cpu()

    # out = [x_hat.detach().cpu().tolist(), x.detach().cpu().tolist(), loss.item()]
    # print(np.array(out).shape())
    # output.append([x_hat.detach().cpu().tolist(), x.detach().cpu().tolist(), loss.item()])
    output['recon'].append(x_hat.detach().cpu().tolist())
    output['og'].append(x.detach().cpu().tolist())
    output['loss'].append(loss.tolist())

# idxs = funcs.get_example_specs(output['loss'])
MU = data_loader.MU
SIGMA = data_loader.SIGMA
l = data_loader.l
funcs.plot_example_specs(output, MU, SIGMA, l)

# # test

# agn_fluxes = data_loader.load_agn()
# agn_dataset = SpecDataset(agn_fluxes)
# agn_loader = torch.utils.data.DataLoader(agn_dataset)

# test_lossses = funcs.test_agn(agn_loader, model)

