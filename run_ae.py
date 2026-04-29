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

import funcs
from datahandling import H5SpecDataset
import autoencoder as ae
import training
import argparse
import time

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-s', '--early_stop', action='store_false') # type=bool not recommended
    parser.add_argument('-b', '--beta', default=0.0, type=float)
    parser.add_argument('-l', '--learn_rate', default=1e-4, type=float)
    parser.add_argument('-d', '--weight_decay', default=1e-8, type=float)

    parser.set_defaults(activation='ReLU', layers=[{'in': 256,   'out': 64, },], latent=32)

    activation_funcs = parser.add_mutually_exclusive_group()
    activation_funcs.add_argument('-r', '--relu', dest='activation', action='store_const', const='ReLU')
    activation_funcs.add_argument('-t', '--tanh', dest='activation', action='store_const', const='Tanh')
    activation_funcs.add_argument('--leaky', dest='activation', action='store_const', const='LeakyReLU')

    architectures = parser.add_mutually_exclusive_group()
    architectures.add_argument('--layers-1', dest="architectures")



    args = parser.parse_args()

    #####################################################################################################
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

    ###############
    ###############
    TESTING = False
    verb = TESTING

    if TESTING:
        batch_size_train = 2
        batch_size_valid = 1
    else:
        batch_size_train = batch_size_valid = 64
    ###############
    ###############

    #####################################################################################################
    # Load data

    DATA = "test_all_spectra.h5"

    # default is normalised data
    train = H5SpecDataset(DATA, split = "train")
    valid = H5SpecDataset(DATA, split = "validation")

    # # 
    # for nw in [0, 2, 4, 8, 12]:
    #     train_loader = torch.utils.data.DataLoader(train, batch_size=64, num_workers=nw)
    #     start = time.time()
        
    #     for i, data in enumerate(train_loader):
    #         if i > 10: break  # Just test the first few batches
            
    #     end = time.time()
    #     print(f"num_workers: {nw} | Time per 10 batches: {end - start:.4f}s")
    # # 

    if device.type == "cpu":
        if os.environ.get("SLURM_CPUS_PER_TASK") is not None:
            num_workers = 8
    else:
        # if gpu or not cluster
        num_workers = 0
        

    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size_train, shuffle = True, num_workers = num_workers)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size_valid, shuffle = False,)

    #################################################################################################


    # intiate test paramters and stuff

    INPUT_SIZE = train[0][0].shape[0]
    CONFIG = [
        {'in': 256,   'out': 64, },
    ]
    LATENT_SIZE = 32
    ACTIVATION_FUNCTION = args.activation

    EPOCHS = args.epochs
    EARLY_STOPPING = args.early_stop
    BETA = args.beta # kl weighting only used in VAE, automatically set as 0 for other models
    LEARNING_RATE = args.learn_rate
    WEIGHT_DECAY = args.weight_decay

    #####################################
    ############# make model ############
    #####################################
    model = ae.StandardAutoencoder(CONFIG, INPUT_SIZE, LATENT_SIZE, activation = ACTIVATION_FUNCTION)
    #####################################

    TEST_NAME = f'RUN_{model.type}_nl{len(CONFIG)}_ls{LATENT_SIZE}_e{EPOCHS}_{ACTIVATION_FUNCTION}_B{BETA:.0e}_lr{LEARNING_RATE:.0e}_wd{WEIGHT_DECAY}_es{EARLY_STOPPING}'

    counter = 1
    base_name = TEST_NAME
    while os.path.isdir(TEST_NAME):
        TEST_NAME = f"{base_name}_{counter}"
        counter += 1


    test_params = {
        'test_name': TEST_NAME,
        'data_file': DATA,
        'ae_type' : model.type,
        'config' : CONFIG,
        'latent_size' : LATENT_SIZE,
        'activation_function' : ACTIVATION_FUNCTION,
        'max_epochs' : EPOCHS,
        'beta' : BETA,
        'learn_rate' : LEARNING_RATE,
        'weight_decay' : WEIGHT_DECAY
    }

    funcs.save_test_params(test_params, TEST_NAME, test=TESTING)

    ######################################################################################################

    print(model)

    if EARLY_STOPPING:
        early_stopping = training.CustomEarlyStopping(TEST_NAME, patience = 10, delta = 0.0, test = TESTING, verbose = verb)
    else:
        early_stopping = None

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    # train
    torch.cuda.empty_cache()
    # model, model_losses = funcs.train_ae(EPOCHS, train_loader, valid_loader, model, optimizer, early_stopping = early_stopping, beta=BETA, verbose = verb, )
    start = time.time()
    trainer = training.Trainer(device, TEST_NAME, model, optimizer, early_stopping, BETA, test = TESTING)
    model, model_losses = trainer.train_ae(EPOCHS, train_loader, valid_loader = valid_loader, verbose = verb)
    stop = time.time()

    print(f"{stop-start} seconds to train")

    funcs.plot_loss(model_losses, test_params['test_name'], test=TESTING)

    funcs.plot_examples(train_loader, model, test_params, test = TESTING)

if __name__ == '__main__':
    
    main()