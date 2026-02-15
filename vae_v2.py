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

LATENT_SIZE = 10

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

class VAEAutoencoder(nn.Module):

    def __init__(self, config):
        super(VAEAutoencoder, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.input_to_encoder = nn.LazyLinear(config[0]['in'])

        # add encoder layers
        for c in config:
            self.encoder_layers.append(
                nn.Linear(c['in'], c['out'], )
            )
        
        # add decoder layers
        for c in reversed(config):
            self.decoder_layers.append(
                nn.Linear(c['out'], c['in'], )
            )
        
        # add latent layers
        self.encoder_to_latent_mean = nn.LazyLinear(LATENT_SIZE)
        self.encoder_to_latent_logvar = nn.LazyLinear(LATENT_SIZE)
        
        self.decoder_from_latent = nn.LazyLinear(config[-1]['out'])

        self.decoder_to_output = nn.LazyLinear(INPUT_SIZE)


    def forward(self, x):
        
        # print(x.shape)

        x = torch.relu(self.input_to_encoder(x))
        # print(x.shape)

        for l in self.encoder_layers:
            x = torch.relu(l(x))
            # print(x.shape)

        mu = torch.relu(self.encoder_to_latent_mean(x))
        logvar = torch.relu(self.encoder_to_latent_logvar(x))
        # print(mu.shape)
        # print(logvar.shape)

        epsilon = torch.randn_like(logvar)
        z = mu + logvar*epsilon # latent of VAE
        # print(z.shape)

        z = torch.relu(self.decoder_from_latent(z))

        for l in self.decoder_layers:
            z = torch.relu(l(z))
            # print(z.shape)

        x_hat = torch.relu(self.decoder_to_output(z))
        # print(x_hat.shape)

        return x_hat, mu, logvar

############################################################################
# main

test_config = [
    {'in': 128,   'out': 64, },
    {'in': 64,  'out': 32, },
    {'in': 32,  'out': 16, },
]

model = VAEAutoencoder(test_config)
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
recon_loss = nn.MSELoss()

def kl_divergence(mu, logvar):

    return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# training and stuff
epochs = 5
# outputs = []
losses1_train = [0] * epochs
losses1_valid = [0] * epochs
# losses2_train = [0] * epochs
# losses2_valid = [0] * epochs
train_losses = [0] * epochs

# model.to(device)

# train
for epoch in range(epochs):
    model.train() # set mode to train
    train_loss = 0

    for specs, _ in train:
        # print(specs.shape)
        # specs = specs.view(-1, INPUT_SIZE).to(device) # .view restores original shape 
        specs = specs.to(device)
        # print(specs.shape)

        reconstructed, mu, logvar = model(specs) # prediction
        # print(reconstructed.shape)

        loss1 = recon_loss(reconstructed, specs) + kl_divergence(mu, logvar) # (avg) loss of batch
        # loss2 = loss_function2(reconstructed, specs)
        optimizer.zero_grad()
        # which loss to base weight updates on
        loss1.backward()
        # loss2.backward()
        # print(train_loss)
        train_loss += loss1.item() # cumulative loss
        # print(train_loss)
        # print(losses1_train[epoch])
        # losses1_train[epoch] += loss1.item()
        # print(losses1_train[epoch])
        # losses1_train[epoch] += loss1.item()
        # losses2_train[epoch] += loss2.item()*specs.size(0)

        optimizer.step()
    
    avg_epoch_loss = train_loss / len(train.dataset)
    print(f'training: epoch {epoch+1}/{epochs}, loss: {avg_epoch_loss:.10f}')
    # losses1_train[epoch] /= len(train.dataset) # sort of weighted average
    # losses2_train[epoch] /= len(train.dataset) # sort of weighted average
    # print(f"TRAINING: Epoch {epoch+1}/{epochs}, LOSS: {loss1.item():.10f}")
    # print(f"TRAINING: Epoch {epoch+1}/{epochs}, Loss: {loss2.item():.10f}")

    # outputs.append((epoch, specs, reconstructed))

    # model.eval() # test mode

    # with torch.no_grad(): # makesure no gradients are calculated
    #     for specs, _ in valid:
    #         specs = specs.view(-1, INPUT_SIZE).to(device) # .view restores original shape 

    #         reconstructed = model(specs) # (prediction)

    #         loss1 = loss_function1(reconstructed, specs)

    #         # loss2 = loss_function2(specs,reconstructed, mean, logvar)
            
    #         # losses.append(loss.item())
    #         losses1_valid[epoch] += loss1.item()*specs.size(0)
    #         # losses2_valid[epoch] += loss2.item()*specs.size(0)
        
    #     losses1_valid[epoch] /= len(valid.dataset) # sort of weighted average
    #     # losses2_valid[epoch] /= len(train.dataset) # sort of weighted average
    #     print(f"VALID: Epoch {epoch+1}/{epochs}, Loss: {loss1.item():.10f}")
    #     # print(f"VALID: Epoch {epoch+1}/{epochs}, Loss: {loss2.item():.10f}")

model.eval()