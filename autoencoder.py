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

import logging

logger = logging.getLogger(__name__)

####### autoencoders ########


class StandardAutoencoder(nn.Module):
	def __init__(self, config, input_size, latent_size, flux_type, normalize, activation="ReLU"):
		super(StandardAutoencoder, self).__init__()

		self.type = "sae"

		self.flux_type = flux_type
		self.normalize = normalize

		self.mean = None
		self.std = None

		self.act_func = getattr(nn, activation)()  # make instance of desired activation function

		self.encoder_layers = nn.ModuleList()
		self.decoder_layers = nn.ModuleList()


		self.input_to_encoder = nn.Linear(input_size, config[0]["in"])

		# add encoder layers
		for c in config:
			self.encoder_layers.append(
				nn.Linear(
					c["in"],
					c["out"],
				)
			)

		self.encoder_to_latent = nn.Linear(config[-1]["out"], latent_size)

		# add decoder layers
		for c in reversed(config):
			self.decoder_layers.append(
				nn.Linear(
					c["out"],
					c["in"],
				)
			)

		self.decoder_from_latent = nn.Linear(latent_size, config[-1]["out"])

		self.decoder_to_output = nn.Linear(config[0]["in"], input_size)

	def forward(self, x):

		# a forward pass

		x = self.act_func(self.input_to_encoder(x))

		for l in self.encoder_layers:
			x = self.act_func(l(x))

		z = self.act_func(self.encoder_to_latent(x))

		z = self.act_func(self.decoder_from_latent(z))

		for l in self.decoder_layers:
			z = self.act_func(l(z))

		x_hat = self.decoder_to_output(z)

		return x_hat, None, None

	def encode(self, x):
		
		x = self.act_func(self.input_to_encoder(x))

		for l in self.encoder_layers:
			x = self.act_func(l(x))

		return self.encoder_to_latent(x)

class VAEAutoencoder(nn.Module):
	def __init__(self, config, input_size, latent_size, flux_type, normalize, activation="ReLU"):
		super(VAEAutoencoder, self).__init__()

		self.type = "vae"

		self.flux_type = flux_type
		self.normalize = normalize

		self.act_func = getattr(
			nn, activation
		)()  # make instance of desired activation function

		self.encoder_layers = nn.ModuleList()
		self.decoder_layers = nn.ModuleList()

		self.input_to_encoder = nn.Linear(input_size, config[0]["in"])

		# add encoder layers
		for c in config:
			self.encoder_layers.append(
				nn.Linear(
					c["in"],
					c["out"],
				)
			)

		# ###### this is cool- remember for future
		# def _get_flattened_size(self, input_size):

		# with torch.no_grad(): # do not update weights

		# dummy_x = torch.zeros(1, 1, input_size)
		# for l in self.encoder_layers:
		# dummy_x = l(dummy_x) # updates the dummy shape based on the encoder layers

		# return dummy_x.numel(), dummy_x.shape[1], dummy_x.shape[2]
		# ######

		# add decoder layers
		for c in reversed(config):
			self.decoder_layers.append(
				nn.Linear(
					c["out"],
					c["in"],
				)
			)

		# add latent layers
		self.encoder_to_latent_mean = nn.Linear(config[-1]["out"], latent_size)
		self.encoder_to_latent_logvar = nn.Linear(config[-1]["out"], latent_size)

		self.decoder_from_latent = nn.Linear(latent_size, config[-1]["out"])

		self.decoder_to_output = nn.Linear(config[0]["in"], input_size)

	# def encode(self, x):
	# 	x = self.act_func(self.input_to_encoder(x))
	# 	for l in self.encoder_layers:
	# 		x = self.act_func(l(x))
	# 	return self.encoder_to_latent_mean(x)

	def forward(self, x):

		x = self.act_func(self.input_to_encoder(x))

		for l in self.encoder_layers:
			x = self.act_func(l(x))

		mu = self.encoder_to_latent_mean(x)
		logvar = self.encoder_to_latent_logvar(x)

		std = torch.exp(0.5 * logvar)
		epsilon = torch.randn_like(std)
		z = mu + std * epsilon  # latent of VAE

		z = self.act_func(self.decoder_from_latent(z))

		for l in self.decoder_layers:
			z = self.act_func(l(z))

		x_hat = self.decoder_to_output(z)

		return x_hat, mu, logvar
	
	def encode(self, x):
		
		x = self.act_func(self.input_to_encoder(x))

		for l in self.encoder_layers:
			x = self.act_func(l(x))

		return self.encoder_to_latent(x)


# class CNNAutoencoder(nn.Module):

# def __init__(self, config):
# super(CNNAutoencoder, self).__init__()

# self.type = 'cnn'

# self.act_func = getattr(nn, activation)() # make instance of desired activation function

# self.encoder_layers = nn.ModuleList()
# self.decoder_layers = nn.ModuleList()

# encoder = []

# in_channels = 1
# for e in encoder:
# self.encoder_layers.append(e)
