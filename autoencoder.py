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
	def __init__(self, config, input_size, latent_size, activation="ReLU"):
		super(StandardAutoencoder, self).__init__()

		self.type = "sae"

		# self.flux_type = flux_type

		# self.mean = None
		# self.std = None

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


	def encode(self, x):

		x = self.act_func(self.input_to_encoder(x))

		for l in self.encoder_layers:
			x = self.act_func(l(x))

		# return self.act_func(self.encoder_to_latent(x))
		return self.encoder_to_latent(x)

	def decode(self, x):

		x = self.act_func(self.decoder_from_latent(x))

		for l in self.decoder_layers:
			x = self.act_func(l(x))

		return self.decoder_to_output(x)


	def forward(self, x):

		# a forward pass

		z = self.encode(x)

		return self.decode(z), None, None


class VAEAutoencoder(nn.Module):
	def __init__(self, config, input_size, latent_size, activation="ReLU"):
		super(VAEAutoencoder, self).__init__()

		self.type = "vae"

		# self.flux_type = flux_type

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

	def encode(self, x):
		# for tsne/umap
		h = self._encoder_trunk(x)
		return self.encoder_to_latent_mean(h)

	def _encoder_trunk(self, x):
		x = self.act_func(self.input_to_encoder(x))
		for l in self.encoder_layers:
			x = self.act_func(l(x))
		return x

	def decode(self, z):
		z = self.act_func(self.decoder_from_latent(z))
		for l in self.decoder_layers:
			z = self.act_func(l(z))
		return self.decoder_to_output(z)

	def forward(self, x):
		h = self._encoder_trunk(x)
		mu     = self.encoder_to_latent_mean(h)
		logvar = self.encoder_to_latent_logvar(h)

		# clamp logvar to avoid nans
		logvar = torch.clamp(logvar, min=-10.0, max=10.0)

		std = torch.exp(0.5 * logvar)
		z = mu + std * torch.randn_like(std)     # reparameterised sample
		return self.decode(z), mu, logvar

def build_model(params, state_dict=None, device = "cpu"):

	# rebuild model from params.json

	cls = {
		"StandardAutoencoder": StandardAutoencoder,
		"VariationalAutoencoder": VAEAutoencoder,
	}[params["ae_type"]]

	model = cls(
		params["config"],
		params["input_size"],
		params["latent_size"],
		activation=params["activation_function"],
	)

	if state_dict is not None:
		model.load_state_dict(state_dict)

	return model.to(device).eval()

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
