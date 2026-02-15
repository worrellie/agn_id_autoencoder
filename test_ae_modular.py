# autoencoder with spectral data, modular design


import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import torch.nn.functional as F
from torch.distributions.normal import Normal

spec_dir = "/home/worrellie/Documents/phd/autoencoder/Datasets/z08_v3-002/"
# spec_path = os.path.join(spec_dir, spec_name)
fluxes = []
for spec in os.listdir(spec_dir):
    if spec.endswith("1h_RI.fits"):
        spec_path = os.path.join(spec_dir, spec)
        try:
            with fits.open(spec_path) as hdul:

                flux = hdul[1].data
                flux = flux.astype(np.float32)
                flux = torch.from_numpy(flux)
                fluxes.append(flux)

        except Exception as e:
            print(f"Error opening spectrum: {spec} ({e})")

fluxes = np.asarray(fluxes)
fluxes = torch.from_numpy(fluxes)
print(fluxes.size())

mu = fluxes.mean(dim=0, keepdim=True)
sigma = fluxes.std(dim=0, keepdim=True)

fluxes_std = (fluxes - mu) / sigma # standardized fluxes

class SpecDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data[idx]

        return sample, sample

class SpectraDataset(torch.utils.data.Dataset):

    def __init__(self, spec_dir, transform = None,):
        self.spec_dir = spec_dir
        self.spec_list = [s for s in os.listdir(spec_dir) if s.endswith('1h_RI.fits')]
        self.transform = transform
    
    def __len__(self):
        
        return len(self.spec_list)

    def __getitem__(self, idx):
        spec_name  = self.spec_list[idx]
        spec_path = os.path.join(self.spec_dir, spec_name)

        # open fits and get necessary data: (for now, just l and f)
        try:
            with fits.open(spec_path) as hdul:
                # print(hdul[1].header['CRVAL1']) # 6469.999999999999
                # print(hdul[1].header['CDELT1']) # 0.6970899470899469

                flux = hdul[1].data
                flux = flux.astype(np.float32)

                sample = torch.from_numpy(flux)

            if self.transform:
                self.transform(sample)
            
            # needs to return 'image' and target. since training AE, target is the input.
            return sample, sample

        except Exception as e:
            print(f"Error opening spectrum: {spec_name} ({e})")

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.flatten = nn.Flatten()

        # define layers
        self.linear1 = nn.Linear(4117, 128) # input to first hidden
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 36)
        self.linear4 = nn.Linear(36, 18)
        self.linear5 = nn.Linear(18, 9)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)

        encoded = x

        return encoded

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(9, 18)
        self.linear2 = nn.Linear(18, 36)
        self.linear3 = nn.Linear(36, 64)
        self.linear4 = nn.Linear(64, 128)
        self.linear5 = nn.Linear(128, 4117) # output layer

    def forward(self, z):

        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = self.linear5(z)

        # z = torch.sigmoid(z)
        decoded = z.view(-1, 4117)

        return decoded
    
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):

        z = self.encoder(x)
        y = self.decoder(z)

        return y



############################################################################
# main

# spec_data = SpectraDataset("/home/worrellie/Documents/phd/autoencoder/Datasets/z08_v3-002/", )
spec_data = SpecDataset(fluxes_std)

loader = torch.utils.data.DataLoader(spec_data, batch_size=64, shuffle=True)

samples, _ = next(iter(loader))
# print(samples.size())

# make instance of autoencoder
model = Autoencoder()
# loss function
loss_function = nn.MSELoss()
# optimization w/ Adam, preventing overfitting (how?)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# training and stuff
epochs = 50
outputs = []
losses = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):

    for specs, _ in loader:
        specs = specs.view(-1, 4117).to(device) # .view restores original shape 
        # specs = specs.to(device)
        reconstructed = model(specs)
        # print(reconstructed)
        # reconstructed = reconstructed.view(-1, 5742, 2) # .view restores original shape

        loss = loss_function(reconstructed, specs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    outputs.append((epoch, specs, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.10f}")

# loss curve
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
# plt.show()

# visualising original and reconstructed images
model.eval()
dataiter = iter(loader)
specs, _ = next(dataiter)

specs = specs.to(device)
reconstructed = model(specs)

spec = specs[0].cpu().detach().numpy()
recon = reconstructed[0].cpu().detach().numpy()
l = np.linspace(6469.9999, 9339.9193, num=4117)

# plot first spec and reconstruction
fig, ax = plt.subplots()
ax.plot(l, spec, label = 'spec', lw=1)
ax.plot(l, recon, label = ' recon', lw = 1)
plt.legend()
plt.show()