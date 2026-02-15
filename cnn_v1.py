# CNN autoencoder with spectral data, modular design and validation set

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

mu = fluxes.mean()
sigma = fluxes.std()

fluxes_std = (fluxes - mu) / sigma # standardized fluxes

f_train, f_test = train_test_split(fluxes_std)
f_train, f_valid = train_test_split(f_train, test_size = 0.1)

f_train = np.asarray(f_train)
f_test = np.asarray(f_test)
f_valid = np.asarray(f_valid)

f_train = torch.from_numpy(f_train)
f_valid = torch.from_numpy(f_valid)
f_test = torch.from_numpy(f_test)



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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=None, padding=0)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=None, padding=0)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

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
train_dataset = SpecDataset(f_train)
valid_dataset = SpecDataset(f_valid)
test_dataset = SpecDataset(f_test)

torch.manual_seed(42)

train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# samples, _ = next(iter(loader))
# print(samples.size())

# make instance of autoencoder
model = Autoencoder()
print(model)
# loss function
loss_function = nn.MSELoss()
# optimization w/ Adam, preventing overfitting (how?)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# training and stuff
epochs = 50
outputs = []
losses_train = [0] * epochs
losses_valid = [0] * epochs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train() # set mode to train

    for specs, _ in train:
        specs = specs.view(-1, 4117).to(device) # .view restores original shape 

        reconstructed = model(specs) # (prediction)

        loss = loss_function(reconstructed, specs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # losses.append(loss.item())
        losses_train[epoch] += loss.item()*specs.size(0)
    
    losses_train[epoch] /= len(train.dataset) # sort of weighted average
    outputs.append((epoch, specs, reconstructed))
    print(f"TRAINING: Epoch {epoch+1}/{epochs}, Loss: {loss.item():.10f}")

    model.eval()

    with torch.no_grad(): # makesure no gradients are calculated
        for specs, _ in valid:
            specs = specs.view(-1, 4117).to(device) # .view restores original shape 

            reconstructed = model(specs) # (prediction)

            loss = loss_function(reconstructed, specs)

            # losses.append(loss.item())
            losses_valid[epoch] += loss.item()*specs.size(0)

    losses_valid[epoch] /= len(valid.dataset) # sort of weighted average
    print(f"VALIDATION: Epoch {epoch+1}/{epochs}, Loss: {loss.item():.10f}")

model.eval()
###################################################
# loss curve
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses_train, label='Train')
plt.plot(losses_valid, label='Valid')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

###################################################

# test data

###################################################

# visualising original and reconstructed images
model.eval()
dataiter = iter(train)
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