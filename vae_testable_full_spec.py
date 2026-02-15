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

LATENT_SIZE = 128

################### galaxies ###################

spec_dir = "/home/worrellie/Documents/phd/autoencoder/merged_spectra_gal"
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

################## classes ###################

class SpecDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data[idx]

        return sample, sample

class Encoder(nn.Module):

    def __init__(self, n_layers, n_input, latent_size, ):
        super(Encoder, self).__init__()

        self.flatten = nn.Flatten()

        # define layers
        self.n_layers = n_layers
        self.n_input = n_input
        self.latent_size = latent_size
        self.n_output = self.latent_size * 3
        o1 = int(self.n_output/(np.pow(0.5, (self.n_layers-1))))
        # print(o1)

        encoder_layers = [0] * self.n_layers

        for l in range(len(encoder_layers)):
            if l == 0:
                input = self.n_input
                output = o1
            # print(l, input, output)
            encoder_layers[l] = nn.Linear(input, output)
            input = output
            output = int(input/2)
        
        # must be ModuleList for PyTorch to see/read
        self.encoder_layers = nn.ModuleList(encoder_layers) 

        self.latent_mean = nn.Linear(self.n_output, self.latent_size)
        self.latent_logvar = nn.Linear(self.n_output, self.latent_size)

    def forward(self, x):

        x = self.flatten(x)

        for layer in self.encoder_layers:
            x = F.relu(layer(x))

        mean, logvar = self.latent_mean(x), self.latent_logvar(x)

        encoded = x # where the encoded/ latent layer is statistically produced

        return mean, logvar

class Decoder(nn.Module):

    def __init__(self, n_layers, latent_size, n_output, ):
        super(Decoder, self).__init__()

        # define layers
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.n_output = n_output

        decoder_layers = [0] * (self.n_layers+1)

        for l in range(len(decoder_layers)-1):
            if l == 0:
                input = self.latent_size
                output = self.latent_size * 3
            # print(l, input, output)
            decoder_layers[l] = nn.Linear(input, output)
            input = output
            output = int(input*2)
        input = int(output/2)

        decoder_layers[-1] = nn.Linear(input, self.n_output)
        
        # must be ModuleList for PyTorch to see/read
        self.decoder_layers = nn.ModuleList(decoder_layers)


    def forward(self, z):

        for l in range(len(self.decoder_layers)-1):
            layer = self.decoder_layers[l]
            z = F.relu(layer(z))

        last_layer = self.decoder_layers[-1]
        z = last_layer(z)

        decoded = z.view(-1, self.n_output)

        return decoded
    
class Autoencoder(nn.Module):

    def __init__(self, n_layers, n_input, latent_size ):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(n_layers, n_input, latent_size)
        n_output = n_input
        self.decoder = Decoder(n_layers, latent_size, n_output)

    def forward(self, x):

        mean, logvar = self.encoder(x)

        epsilon = torch.randn_like(logvar).to(device)
        z = mean + logvar*epsilon

        x_hat = self.decoder(z)

        return x_hat, mean, logvar

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

n_layers = 5
latent_size = 32
input_size = 12217
print(f'layers: {n_layers}, latent size: {latent_size}')

# make instance of autoencoder
model = Autoencoder(n_layers, input_size, latent_size).to(device)
print(model)
##########################################################################
# loss functions
loss_function1 = nn.MSELoss()

def loss_function2(x, x_hat, mean, logvar):
    # kl divergence
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())

    return reproduction_loss + KLD


###########################################################################
# optimization w/ Adam, preventing overfitting (how?)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# training and stuff
epochs = 30
outputs = []
losses1_train = [0] * epochs
losses1_valid = [0] * epochs
losses2_train = [0] * epochs
losses2_valid = [0] * epochs

model.to(device)

for epoch in range(epochs):
    model.train() # set mode to train

    for specs, _ in train:
        specs = specs.view(-1, 12217).to(device) # .view restores original shape 

        reconstructed, mean, logvar = model(specs) # (prediction)

        loss1 = loss_function1(reconstructed, specs)

        loss2 = loss_function2(specs,reconstructed, mean, logvar)
        
        optimizer.zero_grad()
        # which loss to base weight updates on
        loss1.backward()
        # loss2.backward()
        optimizer.step()
        
        # losses.append(loss.item())
        losses1_train[epoch] += loss1.item()*specs.size(0)
        losses2_train[epoch] += loss2.item()*specs.size(0)
    
    losses1_train[epoch] /= len(train.dataset) # sort of weighted average
    losses2_train[epoch] /= len(train.dataset) # sort of weighted average
    print(f"TRAINING: Epoch {epoch+1}/{epochs}, Loss: {loss1.item():.10f}")
    print(f"TRAINING: Epoch {epoch+1}/{epochs}, Loss: {loss2.item():.10f}")

    outputs.append((epoch, specs, reconstructed))

    model.eval() # test mode

    with torch.no_grad(): # makesure no gradients are calculated
        for specs, _ in valid:
            specs = specs.view(-1, 12217).to(device) # .view restores original shape 

            reconstructed, mean, logvar = model(specs) # (prediction)

            loss1 = loss_function1(reconstructed, specs)

            loss2 = loss_function2(specs,reconstructed, mean, logvar)
            
            # losses.append(loss.item())
            losses1_valid[epoch] += loss1.item()*specs.size(0)
            losses2_valid[epoch] += loss2.item()*specs.size(0)
        
        losses1_valid[epoch] /= len(valid.dataset) # sort of weighted average
        losses2_valid[epoch] /= len(train.dataset) # sort of weighted average
        print(f"VALID: Epoch {epoch+1}/{epochs}, Loss: {loss1.item():.10f}")
        print(f"VALID: Epoch {epoch+1}/{epochs}, Loss: {loss2.item():.10f}")

model.eval()
###################################################

# loss curves
# kl divergence
##
# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(8, 5))
# plt.plot(losses2_train, label='Train')
# plt.plot(losses2_valid, label='Valid')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# mse loss
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses1_train, label='Train')
plt.plot(losses1_valid, label='Valid')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()


###################################################

# reconstruct AGN data as tst for prrof of concept
model.eval()
dataiter = iter(agn)
a, _ = next(dataiter)

loss_f = nn.MSELoss(reduction = 'none') # reconstruction error or ELBO

errors = []
with torch.no_grad(): # Disable gradient calculation to save memory
    for a, _ in agn:
        inputs = a.to(device) # Adjust based on your loader structure
        outputs, _, _ = model(inputs)
        
        # Calculate MSE per sample
        loss = loss_f(outputs, inputs)
        # Flatten and mean per sample: [batch_size, channels, h, w] -> [batch_size]
        sample_loss = loss.view(loss.size(0), -1).mean(dim=1)
        
        errors.extend(sample_loss.cpu().numpy())

# print(errors)
print(min(errors))
print(max(errors))
print(np.mean(errors))

print(len(errors))

short_errs = [e for e in errors if e<=1]
print(len(short_errs))

pc = (len(short_errs)/len(errors))*100
print(f'{pc}%')

plt.figure(figsize=(8, 5))
plt.hist(errors, bins= 200)
# plt.xscale('log')
plt.xlabel('Loss')
plt.ylabel('N')
# plt.yscale('log')
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(short_errs, bins= 200)
plt.xlabel('Loss')
plt.ylabel('N')
plt.legend()
plt.show()



###################################################

# visualising original and reconstructed images
model.eval()
dataiter = iter(train)
specs, _ = next(dataiter)

specs = specs.to(device)
with torch.no_grad():
    output, _, _ = model(specs)

spec = specs[0].cpu().numpy()    # get first spec and move to cpu
recon = output[0].cpu().numpy()

spec = reconstruct(spec)
recon = reconstruct(recon)

l = l

# plot first spec and reconstruction
fig, ax = plt.subplots()
ax.plot(l, spec, label = 'spec', lw=1)
ax.plot(l, recon, label = ' recon', lw = 1)
plt.legend()
plt.show()

