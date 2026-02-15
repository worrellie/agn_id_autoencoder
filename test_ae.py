# autoencoder with spectral data but not good design


import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


class SpectraDataset(torch.utils.data.Dataset):

    def __init__(self, spec_dir, transform = None, target_transform = None):
        self.spec_dir = spec_dir
        self.spec_list = [s for s in os.listdir(spec_dir) if s.endswith('.fits')]
        self.transform = transform
    
    def __len__(self):
        
        return len(self.spec_list)

    def __getitem__(self, idx):
        spec_name  = self.spec_list[idx]
        spec_path = os.path.join(self.spec_dir, spec_name)

        # open fits and get necessary data: (for now, just l and f)
        try:
            with fits.open(spec_path) as hdul:
                data = hdul[1].data
                l= data['l']
                flux = data['flux']
                sample = np.stack([l, flux], axis=1)
                sample = torch.from_numpy(sample.astype(np.float32))

                # flux = hdul[2].data # Obs noNoise
                # l = hdul[9].data # Vacuum wavelengths
                # spec = torch.tensor([flux, l])

            if self.transform:
                self.transform(sample)
            
            # needs to return 'image' and target. since training AE, target is the input.
            return sample, sample

        except Exception as e:
            print(f"Error opening spectrum: {spec_name} ({e})")


# define autoencoder model
class AE(nn.Module):

    def __init__(self):
        
        super(AE, self).__init__()

        #
        self.flatten = nn.Flatten()
        #

        # compress 28x28 pixels into smaller latent representation,
        # uses fully-connected laters with ReLU activations
        self.encoder = nn.Sequential(
            nn.Linear(5742 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9)
        )

        # reconstruct image by expanding latent vector to original size
        # end with sigmoid to output [0,1] values
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 5742 * 2),

            # nn.Sigmoid()
            # nn.Linear()
            
            
        )

    def forward(self, x):

        x = self.flatten(x)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 5742, 2)
        
        return decoded

############################################################################
# main

spec_data = SpectraDataset("/home/worrellie/Documents/phd/autoencoder/new_specs")

loader = torch.utils.data.DataLoader(spec_data, batch_size=5, shuffle=True)

# samples, _ = next(iter(loader))
# print(samples.size())

# exit()

# make instance of autoencoder
model = AE()
# loss function
loss_function = nn.MSELoss()
# optimization w/ Adam, preventing overfitting (how?)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# training and stuff
epochs = 120
outputs = []
losses = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):

    for specs, _ in loader:
        specs = specs.view(-1, 5742, 2).to(device) # .view restores original shape 
        # specs = specs.to(device)

        reconstructed = model(specs)
        # reconstructed = reconstructed.view(-1, 5742, 2) # .view restores original shape

        loss = loss_function(reconstructed, specs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    outputs.append((epoch, specs, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

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

print(min(spec[:,1]))
print(max(spec[:,1]))

print(min(spec[:,0]))
print(max(spec[:,0]))


# plot first spec and reconstruction
print(min(recon[:,1]))
print(max(recon[:,1]))

print(min(recon[:,0]))
print(max(recon[:,0]))


fig, ax = plt.subplots()
ax.plot(spec[:, 0], spec[:, 1], label = 'spec', lw=1)
ax.plot(recon[:, 0], recon[:, 1], label = ' recon', lw = 1)
plt.legend()
plt.show()