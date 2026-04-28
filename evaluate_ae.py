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
import pathlib as path
import h5py
from datahandling import H5SpecDataset
import autoencoder as ae
import funcs
import json
from random import randrange
from matplotlib import pyplot as plt

# get device
print(f"GPU available: {torch.cuda.is_available()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device type: {device.type}")
if device.type == "cpu":
    try:
        num_threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    except:
        num_threads = min(4, os.cpu_count())
        print(f'cannot get SLURM_CPUS_PER_TASK, defaulting to {num_threads}')
    finally:
        torch.set_num_threads(num_threads)
        print(f"num threads set to: {torch.get_num_threads()}")

if device.type == "cpu":
    if os.environ.get("SLURM_CPUS_PER_TASK") is not None:
        num_workers = 8
else:
    # if gpu or not cluster
    num_workers = 0


# load test set
DATA = "all_spectra_to_process_sf.h5"
TEST_DIR = "tests_270426\RUN_sae_nl4_ls32_e1000_ReLU_B0e+00_lr1e-04_wd1e-08_esFalse_1"
MODEL_PATH = os.path.join(TEST_DIR, "RUN_sae_nl4_ls32_e1000_ReLU_B0e+00_lr1e-04_wd1e-08_esFalse_1_model.pt")

test_res_path = path.Path(TEST_DIR, f"test_results.json")
if os.path.exists(test_res_path):
    # load exiting test results
    print('test already predicted. opening existing..')
    with open(test_res_path, 'r') as t:
        test_results = json.load(t)
else:
    # get test results and save
    flux_type = 'normalized_flux'
    if flux_type == 'normalized_flux':
        mean_key = "norm_mean"
        std_key = "norm_std"
    else:
        mean_key = "raw_mean"
        std_key = "raw_std"

    train_mean = None
    train_std = None
    with h5py.File(DATA, 'r') as hf:
        train_mean = hf.attrs[mean_key]
        train_std = hf.attrs[std_key]
    if train_mean is not None and train_std is not None:
        print("normalizing input data")
        normalize = True
    elif train_mean is None and train_std is None:
        normalize = False
    else:
        print('something is wrong wtih the training mean and std..')
        normalize = False

    test = H5SpecDataset(DATA, split = "test")
    test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = True, num_workers = num_workers)

    # load model
    print("loading model")
    model =  torch.load(MODEL_PATH, weights_only = False)
    print(model)

    model.to(device)

    test_results = {
        "xs" : [],
        "x_hats" : [],
        "recon_losses" : [],
        "recon_mean" : None,
        "recon_median" : None
    }

    model.eval()
    with torch.no_grad():

        for x, x_mask in test_loader:

            test_results['xs'].append(x.squeeze().tolist())

            if normalize:
                x = (x - train_mean) / train_std # normalize data
                x = x * x_mask # re-set 'gaps'/masked regions as zero

            x = x.to(device)
            x_mask = x_mask.to(device)

            x_hat, mu, logvar = model(x) # batch prediction

            test_results['x_hats'].append(x_hat.cpu().squeeze().tolist())

            reconstruction_loss = funcs.loss_calc_per_spec(x_hat, x, x_mask, )
            reconstruction_loss.cpu().float()


            test_results['recon_losses'].append(reconstruction_loss.cpu().squeeze().float().item())
    
    test_results['recon_mean'] = np.mean(test_results['recon_losses'])
    test_results['recon_median'] = np.median(test_results['recon_losses'])

    with open(test_res_path, 'w') as t:
        json.dump(test_results, t)

rand_idx = randrange(len(test_results['xs']))
# recon_range = 

# plot rand spec
# plot spec with recon loss closest to given value
# plot 5 spec with recon loss in range (random if there are more than 5)

# plot hist
plt.hist(test_results['recon_losses'], bins = 100)
plt.title(f"{TEST_DIR}\nTest Set Reconstruction Losses\nMean: {test_results['recon_mean']}, Median: {test_results['recon_median']}")
plt.show()
