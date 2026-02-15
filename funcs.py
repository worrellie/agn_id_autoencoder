
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_calc(x_hat, x, mu, logvar):

    recon_loss = nn.MSELoss()

    loss = recon_loss(x_hat, x) + (- 0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

    return loss


def train_ae(epochs, train_loader, valid_loader, model, optimizer, verbose = True, *args):

    model.to(device)

    recon_loss = nn.MSELoss()


    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        valid_loss = 0

        for x, _ in train_loader:

            x = x.to(device)

            x_hat, mu, logvar = model(x) # batch prediction

            loss = loss_calc(x_hat, x, mu, logvar)

            optimizer.zero_grad()

            loss.backward()

            train_loss += loss.item() # cumulative loss

            optimizer.step()

        epoch_avg_loss = train_loss / len(train_loader.dataset) # average loss per sample 
        train_losses.append(epoch_avg_loss) # losses for each epoch

        if verbose:
            print(f'training: epoch {epoch+1}/{epochs}, loss: {epoch_avg_loss:.10f}')

        model.eval()

        with torch.no_grad():

            for x, _ in valid_loader:

                x = x.to(device)

                x_hat, mu, logvar = model(x)

                loss = loss_calc(x_hat, x, mu, logvar)
                
                valid_loss += loss.item()
            
            epoch_avg_valid_loss = valid_loss / len(valid_loader.dataset)
            valid_losses.append(epoch_avg_valid_loss)

        if verbose:
            print(f'valid: epoch {epoch+1}/{epochs}, loss: {epoch_avg_valid_loss:.10f}')


    return model, train_losses, valid_losses

def plot_loss(train_loss, valid_loss):

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train')
    plt.plot(valid_loss, label='Valid')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def test_agn(loader, model):

    model.eval()

    losses = []
    with torch.no_grad():

        recon_loss = nn.MSELoss()

        total_loss = 0
        for x, _ in loader:

            x = x.to(device)

            x_hat, mu, logvar = model(x)

            loss = recon_loss(x_hat, x) + (- 0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader.dataset)
        losses.append(avg_loss)

    print(min(losses))
    print(max(losses))
    print(np.mean(losses))


def unstandardize(spec, MU, SIGMA):

    new_spec = (np.asarray(spec) * SIGMA) + MU

    return new_spec


def _get_example_specs(losses):

    min_i = losses.index(min(losses))

    max_i = losses.index(max(losses))

    mean_i = losses.index(np.median(losses))

    q25_i = losses.index(np.quantile(losses, 0.25))

    q75_i = losses.index(np.quantile(losses, 0.75))

    indices = [min_i, q25_i, mean_i, q75_i, max_i,]
    # print(indices)
    # print([losses[i] for i in indices])

    return indices

def plot_example_specs(output, MU, SIGMA, l):

    indices = _get_example_specs(output['loss'])

    fig = plt.figure(figsize=(20, 7))
    # gs = fig.add_gridspec(2, 6)
    
    # ax1 = fig.add_subplot(gs[0, 0:3])
    ax1 = plt.subplot2grid((2, 6), (0, 1), colspan=2)
    # ax1.set_title("Top Left Plot")

    # ax2 = fig.add_subplot(gs[0, 3:6])
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=2)
    # ax2.set_title("Top Right Plot")

    # ax3 = fig.add_subplot(gs[1, 0:2])
    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    # ax3.set_title("Bottom Left")

    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    # ax4 = fig.add_subplot(gs[1, 2:4])
    # ax4.set_title("Bottom Middle")

    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    # ax5 = fig.add_subplot(gs[1, 4:6])
    # ax5.set_title("Bottom Right")

    axes = [ax1, ax3, ax4, ax5, ax2]

    plt.tight_layout()


    for i, ax in zip(indices, axes):
        # print(i, ax)

        reconstructed = output['recon'][i]
        # reconstructed = unstandardize(reconstructed, MU, SIGMA)
        og = output['og'][i]
        # og = unstandardize(og, MU, SIGMA)
        loss = output['loss'][i]
        ax.plot(l, og, color='black')
        # ax.plot(l, reconstructed, color = 'red')
        ax.set_title(loss)

    plt.show()


    return 