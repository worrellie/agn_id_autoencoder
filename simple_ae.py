# autoencoder with test data


import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# load dataset of images and convert them to tensors 
tensor_transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(root="./data", train=True,
                         download=True, transform=tensor_transform)
loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=1, shuffle=True)

# define autoencoder model
class AE(nn.Module):

    def __init__(self):
        
        super(AE, self).__init__()

        # compress 28x28 pixels into smaller latent representation,
        # uses fully-connected laters with ReLU activations
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
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
            nn.Linear(128, 28 * 28),

            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# make instance of autoencoder
model = AE()
# loss function
loss_function = nn.MSELoss()
# optimization w/ Adam, preventing overfitting (how?)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# training and stuff
epochs = 20
outputs = []
losses = []

# ??
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    for images, _ in loader:
        print(images)
        images = images.view(-1, 28 * 28).to(device)
        
        exit()
        
        reconstructed = model(images)
        loss = loss_function(reconstructed, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    outputs.append((epoch, images, reconstructed))
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
images, _ = next(dataiter)

images = images.view(-1, 28 * 28).to(device)
reconstructed = model(images)
print(images.size())
print(reconstructed.size())

fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))
for i in range(10):
    axes[0, i].imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
# plt.show()