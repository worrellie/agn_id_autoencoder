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

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps") # For Mac users
# else:
#     device = torch.device("cpu")
#     print("Using CPU")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LATENT_SIZE = 128

################### galaxies ###################

spec_dir = "/home/worrellie/Documents/phd/autoencoder/test_gal"
# spec_path = os.path.join(spec_dir, spec_name)
fluxes = []
for spec in os.listdir(spec_dir):
    spec_path = os.path.join(spec_dir, spec)
    try:
        with fits.open(spec_path) as hdul:

            data = hdul[1].data
            flux = data['flux']
            l = data['lambda']
            flux = flux.astype(np.float32)
            flux = torch.from_numpy(flux)
            fluxes.append(flux)

    except Exception as e:
        print(f"Error opening spectrum: {spec} ({e})")

fluxes = np.asarray(fluxes)
INPUT_SIZE = len(fluxes[1])

f_train, f_test = train_test_split(fluxes)
f_train, f_valid = train_test_split(f_train, test_size = 0.1)

MU = float(f_train.mean())   # MU and SIGMA of training only
SIGMA = float(f_train.std()) # otherwise have data leakage

f_train = (f_train - MU) / SIGMA # standardized fluxes of TRAINING ONLY
f_test = (f_test - MU) / SIGMA
f_valid = (f_valid - MU) / SIGMA

f_train = np.asarray(f_train)
f_test = np.asarray(f_test)
f_valid = np.asarray(f_valid)

f_train = torch.from_numpy(f_train)
f_valid = torch.from_numpy(f_valid)
f_test = torch.from_numpy(f_test)

def reconstruct(input):

    reconstructed = (input * SIGMA) + MU

    return reconstructed

################### agn ###################

agn_dir = "/home/worrellie/Documents/phd/autoencoder/agn/merged_spectra_agn"

fluxes_agn = []

for spec in os.listdir(agn_dir):
    spec_path = os.path.join(agn_dir, spec)
    if "z0.9" in spec_path:
        try:
            with fits.open(spec_path) as hdul:
                data = hdul[1].data
                flux = data['flux']
                flux = flux.astype(np.float32)
                flux = torch.from_numpy(flux)
                fluxes_agn.append(flux)

        except Exception as e:
            print(f"Error opening spectrum: {spec} ({e})")

fluxes_agn = np.asarray(fluxes_agn)

fluxes_agn_std = (fluxes_agn - MU) / SIGMA # standardized fluxes

f_agn = np.asarray(fluxes_agn_std)

f_agn = torch.from_numpy(f_agn)

################### conv output #####################

def conv_out_size(n, p, k, s):

    return math.floor(((n + (2*p) - k)/(s)) + 1)

################### dataset #####################

class SpecDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data[idx]

        return sample, sample

################## autoencoder #####################

class CNNAutoencoder(nn.Module):

    def __init__(self, config):
        super(CNNAutoencoder, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # add encoder layers
        for c in config:
            self.encoder_layers.append(
                nn.Conv1d(c['in'], c['out'], c['kernel'], 
                          stride=c['stride'], padding=c['padding'])
            )

        # add decoder layers
        for c in reversed(config):
            self.decoder_layers.append(
                nn.ConvTranspose1d(c['out'], c['in'], c['kernel'], 
                          stride=c['stride'], padding=c['padding'])
            )

        # add latent layers
        self.encoder_to_latent_mean = nn.LazyLinear(LATENT_SIZE)
        self.encoder_to_latent_logvar = nn.LazyLinear(LATENT_SIZE)

    def forward(self, x):

        x_shapes_encoder = []
        x_chans_encoder = []
        x = x.unsqueeze(1) # add channel dimension
        print(x.shape)
        for l in self.encoder_layers:
            x = torch.relu(l(x))
            print(x.shape)
            x_shapes_encoder.append(x.shape[-1])
            x_chans_encoder.append(x.shape[-2])

        x = torch.flatten(x, start_dim = 1)
        flat = x.shape[1]
        print((f'flat: {flat}'))
        print(x.shape)

        mean = torch.relu(self.encoder_to_latent_mean(x))
        logvar = torch.relu(self.encoder_to_latent_logvar(x))
        print(mean.shape)
        print(logvar.shape)

        epsilon = torch.randn_like(logvar).to(device)
        z = mean + logvar*epsilon # latent of VAE
        print(z.shape)

        # print(x_shapes_encoder)

        self.decoder_from_latent = nn.LazyLinear(flat).to(device)
        z = torch.relu(self.decoder_from_latent(z))
        print(z.shape)

        z = z.view(-1, x_chans_encoder[-1], x_shapes_encoder[-1]) # reshape for decoder
        print(z.shape)


        for l in self.decoder_layers:
            z = torch.relu(l(z))
            print(z.shape)

        z = z.squeeze()
        print(z.shape)

        return z

############################################################################
# main

test_config = [
    {'in': 1,   'out': 16,  'kernel': 3, 'stride': 2, 'padding': 1},
    {'in': 16,  'out': 32,  'kernel': 3, 'stride': 2, 'padding': 1},
    {'in': 32,  'out': 64,  'kernel': 3, 'stride': 2, 'padding': 1},
]


model = CNNAutoencoder(test_config, INPUT_SIZE, LATENT_SIZE)
model.to(device)
print(model)

train_dataset = SpecDataset(f_train)
valid_dataset = SpecDataset(f_valid)
test_dataset = SpecDataset(f_test)

train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

#################################################################################
# # test input
# train = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
# test_input = next(iter(train))
# test_input = test_input[0]
# test_input = test_input.to(device)
# print(test_input.shape)

# model(test_input)

# exit()

###########################################################################
# optimization w/ Adam, preventing overfitting (how?)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# loss functions
loss_function1 = nn.MSELoss()

# training and stuff
epochs = 30
outputs = []
losses1_train = [0] * epochs
losses1_valid = [0] * epochs
# losses2_train = [0] * epochs
# losses2_valid = [0] * epochs

# model.to(device)

for epoch in range(epochs):
    model.train() # set mode to train

    for specs, _ in train:
        print(specs.shape)
        specs = specs.view(-1, INPUT_SIZE).to(device) # .view restores original shape 
        print(specs.shape)

        reconstructed = model(specs) # (prediction)

        loss1 = loss_function1(reconstructed, specs)

        # loss2 = loss_function2(specs,reconstructed, mean, logvar)
        
        optimizer.zero_grad()
        # which loss to base weight updates on
        # loss1.backward()
        # loss2.backward()
        optimizer.step()
        
        # losses.append(loss.item())
        losses1_train[epoch] += loss1.item()*specs.size(0)
        # losses2_train[epoch] += loss2.item()*specs.size(0)
    
    losses1_train[epoch] /= len(train.dataset) # sort of weighted average
    # losses2_train[epoch] /= len(train.dataset) # sort of weighted average
    print(f"TRAINING: Epoch {epoch+1}/{epochs}, Loss: {loss1.item():.10f}")
    # print(f"TRAINING: Epoch {epoch+1}/{epochs}, Loss: {loss2.item():.10f}")

    outputs.append((epoch, specs, reconstructed))

    model.eval() # test mode

    with torch.no_grad(): # makesure no gradients are calculated
        for specs, _ in valid:
            specs = specs.view(-1, INPUT_SIZE).to(device) # .view restores original shape 

            reconstructed = model(specs) # (prediction)

            loss1 = loss_function1(reconstructed, specs)

            # loss2 = loss_function2(specs,reconstructed, mean, logvar)
            
            # losses.append(loss.item())
            losses1_valid[epoch] += loss1.item()*specs.size(0)
            # losses2_valid[epoch] += loss2.item()*specs.size(0)
        
        losses1_valid[epoch] /= len(valid.dataset) # sort of weighted average
        # losses2_valid[epoch] /= len(train.dataset) # sort of weighted average
        print(f"VALID: Epoch {epoch+1}/{epochs}, Loss: {loss1.item():.10f}")
        # print(f"VALID: Epoch {epoch+1}/{epochs}, Loss: {loss2.item():.10f}")

model.eval()