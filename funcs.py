
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

def loss_calc(x_hat, x, mu, logvar, red = 'mean'):

    recon_loss = nn.MSELoss(reduction = red)

    loss = recon_loss(x_hat, x) + (- 0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

    return loss


def train_ae(epochs, train_loader, valid_loader, model, optimizer, verbose = True, *args):

    print('training model...')

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

            loss = loss_calc(x_hat, x, mu, logvar,)
            # print(loss)
            # print('------')

            optimizer.zero_grad()

            loss.backward()

            train_loss += loss.item() # cumulative loss

            optimizer.step()

        epoch_avg_loss = train_loss / len(train_loader) # average loss per sample 
        train_losses.append(epoch_avg_loss) # losses for each epoch

        if verbose:
            print(f'training: epoch {epoch+1}/{epochs}, loss: {epoch_avg_loss:.10f}')

        model.eval()
        if valid_loader is not None:

            with torch.no_grad():

                for x, _ in valid_loader:

                    x = x.to(device)

                    x_hat, mu, logvar = model(x)

                    loss = loss_calc(x_hat, x, mu, logvar)

                    
                    valid_loss += loss.item()
                
                epoch_avg_valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(epoch_avg_valid_loss)

            if verbose:
                print(f'valid: epoch {epoch+1}/{epochs}, loss: {epoch_avg_valid_loss:.10f}')
        
    print('training finished')

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


def _get_example_specs(loader, model):

    print('predicting...')
    losses = []
    with torch.no_grad():
        for x, _ in loader:
            # print(x.size())
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss = loss_calc(x_hat, x, mu, logvar, red='none').detach().cpu()
            for l in loss:
                mean_loss = np.mean(np.array(l))
                losses.append(mean_loss)
            # print(len(loss))
            # print(loss)
            # print(loss.item())
            # losses.extend(loss)
    # print(len(losses))

    to_find = {
        "min": np.min(losses),
        "max": np.max(losses),
        "25th": np.percentile(losses, 25),
        "mean": np.mean(losses),
        "75th": np.percentile(losses, 75)
    }

    print('getting min, max, mean and quartiles of losses...')

    idxs = []
    for key, value in to_find.items():
        # print(len(np.abs(losses - value)))
        idx = (np.abs(losses - value)).argmin()
        idxs.append(idx)
    # print(idxs)
    
    # print(to_find)

    return idxs

def _predict_examples(dataset, indices, model):

    output = {'recon' : [],
              'original' : [],
              'loss' : []}

    print('predicting min, max, mean and quartiles...')

    with torch.no_grad():
        for i in indices:
            x = dataset[i][0].to(device)
            # print(type(x))
            # print(x.shape)
            x= x.unsqueeze(0)
            # note: need batch dimension for model
            x_hat, mu, logvar = model(x)
            # print(x_hat.shape)
            loss = loss_calc(x_hat, x, mu, logvar, red='none')
            for l in loss:
                l = l.detach().cpu()
                mean_loss = np.mean(np.array(l))
            x = x.squeeze(0)
            x_hat = x_hat.squeeze(0) # add then remove batch dimension
            x_hat = x_hat.detach().cpu()
            output['recon'].append(x_hat.tolist())
            output['original'].append(x.tolist())
            output['loss'].append(mean_loss)

    # print(len(output['recon']))
    # print(len(output['original']))
    # print(output['loss'])

    return output

def unstandardize(reconstructed, std, n1, n2):

    # n1 is MU or MIN
    # n2 is SIGMA or MAX

    if std == 'zscore':
        recon = (np.array(reconstructed) * n2) + n1
    elif std == 'minmax':
        n = n2 - n1
        # print(type(n))
        # print(type(n1))
        # print(type(n2))
        # print(type(reconstructed))
        recon = (np.array(reconstructed) * n) + n1

    return list(recon)

def _plot_example_specs(output, l, indices, std, n1, n2):

    print('plotting...')

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

    axes = [ax1, ax2, ax3, ax4, ax5]

    plt.tight_layout()


    for i, ax in enumerate(axes):
        # print(i, ax)

        reconstructed = output['recon'][i]
        # print(reconstructed)
        # exit()
        reconstructed = unstandardize(reconstructed, std, n1, n2)
        og = output['original'][i]
        # og = unstandardize(og, MU, SIGMA)
        loss = output['loss'][i]
        ax.plot(l, og, color='black')
        ax.plot(l, reconstructed, color = 'red')
        ax.set_title(loss)

    plt.show()

    return

def plot_examples(loader, model, l, std, n1, n2):

    indices = _get_example_specs(loader, model)

    data = loader.dataset

    output = _predict_examples(data, indices, model)

    _plot_example_specs(output, l, indices, std, n1, n2)

