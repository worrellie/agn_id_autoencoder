# example main
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import datasets
import data_utils
import torch
import utils
import vae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fluxes, l = data_utils.get_raw_data()
# fluxes, l = data_utils.generate_sine_data(1000, 64)

f_train, f_test = train_test_split(fluxes, test_size = 0.25)
f_train, f_valid = train_test_split(f_train, test_size = 0.1)

# standard scaler
std_scaler = preprocessing.StandardScaler()

# minmax scaler
# minmax_scaler = preprocessing.MinMaxScaler()

# robust scaler (robust to outliers)
# robust_scaler = preprocessing.RobustScaler()

train_scaled = std_scaler.fit_transform(f_train)
valid_scaled = std_scaler.transform(f_valid)
test_scaled = std_scaler.transform(f_test)

train_scaled = torch.from_numpy(train_scaled)
valid_scaled = torch.from_numpy(valid_scaled)
test_scaled = torch.from_numpy(test_scaled)

train_dataset = datasets.SpecDataset(train_scaled)
valid_dataset = datasets.SpecDataset(valid_scaled)
test_dataset = datasets.SpecDataset(test_scaled)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

test_config = [
    {'in': 5000,   'out': 2000, },
    {'in': 2000,  'out': 1000, },
    {'in': 1000,  'out': 500, },
    {'in': 500,  'out' : 256, },
]

INPUT_SIZE = len(f_train[1])
LATENT_SIZE = 128

model = vae.VAEAutoencoder(test_config, INPUT_SIZE, LATENT_SIZE)
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

EPOCHS = 50

model, train_losses, valid_losses = utils.train_ae(EPOCHS, train_loader, test_loader, model, optimizer,  verbose = True, )


