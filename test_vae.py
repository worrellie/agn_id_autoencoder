# variational autoencoder with spectral data, modular design and validation set

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

################### agn ###################

agn_dir = "/home/worrellie/Documents/phd/autoencoder/agn/1h_RI/"

fluxes_agn = []

for spec in os.listdir(agn_dir):
    spec_path = os.path.join(agn_dir, spec)
    if "z0.9" in spec_path:
        try:
            with fits.open(spec_path) as hdul:
                flux = hdul[1].data
                flux = flux.astype(np.float32)
                flux = torch.from_numpy(flux)
                fluxes_agn.append(flux)

        except Exception as e:
            print(f"Error opening spectrum: {spec} ({e})")

fluxes_agn = np.asarray(fluxes_agn)

fluxes_agn_std = (fluxes_agn - mu) / sigma # standardized fluxes

f_agn = np.asarray(fluxes_agn_std)

f_agn = torch.from_numpy(f_agn)

##############################################

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

        self.latent_mean = nn.Linear(9, 2)
        self.latent_var = nn.Linear(9, 2)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))

        mean, var = self.latent_mean(x), self.latent_var(x)

        encoded = x # where the encoded/ latent layer is statistically produced

        return mean, var

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.linear0 = nn.Linear(2, 9)

        self.linear1 = nn.Linear(9, 18)
        self.linear2 = nn.Linear(18, 36)
        self.linear3 = nn.Linear(36, 64)
        self.linear4 = nn.Linear(64, 128)
        self.linear5 = nn.Linear(128, 4117) # output layer


    def forward(self, z):

        z = F.relu(self.linear0(z))
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

        mean, var = self.encoder(x)

        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon

        x_hat = self.decoder(z)

        return x_hat, mean, var



############################################################################
# main

# spec_data = SpectraDataset("/home/worrellie/Documents/phd/autoencoder/Datasets/z08_v3-002/", )
train_dataset = SpecDataset(f_train)
valid_dataset = SpecDataset(f_valid)
test_dataset = SpecDataset(f_test)

agn_dataset = SpecDataset(f_agn)

torch.manual_seed(42)

train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

agn = torch.utils.data.DataLoader(agn_dataset, batch_size=16, shuffle=True)

# samples, _ = next(iter(loader))
# print(samples.size())

# make instance of autoencoder
model = Autoencoder().to(device)

##########################################################################
# loss functions
loss_function1 = nn.MSELoss()

def loss_function2(x, x_hat, mean, var):
    # kl divergence
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

###########################################################################
# optimization w/ Adam, preventing overfitting (how?)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# training and stuff
epochs = 50
outputs = []
losses1_train = [0] * epochs
losses1_valid = [0] * epochs
losses2_train = [0] * epochs
losses2_valid = [0] * epochs

model.to(device)

for epoch in range(epochs):
    model.train() # set mode to train

    for specs, _ in train:
        specs = specs.view(-1, 4117).to(device) # .view restores original shape 

        reconstructed, mean, var = model(specs) # (prediction)

        loss1 = loss_function1(reconstructed, specs)

        # loss2 = loss_function2(specs,reconstructed, mean, var)
        
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        
        # losses.append(loss.item())
        losses1_train[epoch] += loss1.item()*specs.size(0)
        # losses2_train[epoch] += loss2.item()*specs.size(0)
    
    losses1_train[epoch] /= len(train.dataset) # sort of weighted average
    outputs.append((epoch, specs, reconstructed))
    # losses2_train[epoch] /= len(train.dataset) # sort of weighted average
    print(f"TRAINING: Epoch {epoch+1}/{epochs}, Loss: {loss1.item():.10f}")

    model.eval()

    with torch.no_grad(): # makesure no gradients are calculated
        for specs, _ in valid:
            specs = specs.view(-1, 4117).to(device) # .view restores original shape 

            reconstructed, mean, var = model(specs) # (prediction)

            loss1 = loss_function1(reconstructed, specs)

            # loss2 = loss_function2(specs,reconstructed, mean, var)
            
            
            # losses.append(loss.item())
            losses1_valid[epoch] += loss1.item()*specs.size(0)
            # losses2_valid[epoch] += loss2.item()*specs.size(0)
        
        losses1_valid[epoch] /= len(valid.dataset) # sort of weighted average
        # losses2_valid[epoch] /= len(train.dataset) # sort of weighted average
        print(f"VALID: Epoch {epoch+1}/{epochs}, Loss: {loss1.item():.10f}")

model.eval()
###################################################
# loss curve
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses1_train, label='Train')
plt.plot(losses1_valid, label='Valid')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

###################################################

# test data: AGN
model.eval()
dataiter = iter(agn)
a, _ = next(dataiter)

loss_f = nn.MSELoss(reduction = 'none') # reconstruction error or ELBO

errors = []
with torch.no_grad(): # Disable gradient calculation to save memory
    for a, _ in agn:
        inputs = a.to(device) # Adjust based on your loader structure
        outputs = model(inputs)
        
        # Calculate MSE per sample
        loss = loss_f(outputs, inputs)
        # Flatten and mean per sample: [batch_size, channels, h, w] -> [batch_size]
        sample_loss = loss.view(loss.size(0), -1).mean(dim=1)
        
        errors.extend(sample_loss.cpu().numpy())

# print(errors)
print(min(errors))
print(max(errors))
print(np.mean(errors))


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