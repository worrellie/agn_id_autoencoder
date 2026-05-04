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

class H5SpecDataset(torch.utils.data.Dataset):

	def __init__(self, data_path, split, flux_type = 'normalized_flux'):
		self.data_path = data_path
		self.split = split
		self.flux_type = flux_type
		if self.flux_type == 'normalized_flux':
			mean_key = "norm_mean"
			std_key = "norm_std"
		else:
			mean_key = "raw_mean"
			std_key = "raw_std"


		with h5py.File(self.data_path, 'r') as hf:
			self.l = hf.attrs['wavelengths'][:]
			self.mean = hf.attrs[mean_key]
			self.std = hf.attrs[std_key]
			self.len = hf[self.split][self.flux_type].shape[0]
			self.n_pixels = hf[self.split][self.flux_type].shape[1]
		
		self.hf = None

	def __len__(self):

		return self.len
	
	def __getitem__(self, idx):

		# lazy loading. only open h5 file when start accessing it
		
		if self.hf is None:
			self.hf = h5py.File(self.data_path, 'r')

		sample = torch.from_numpy(self.hf[self.split][self.flux_type][idx])
		sample_mask = (sample != 0)

		# make sure sample is float32 (best for Pytorch, also I think what is in the h5)
		sample = sample.float()
		sample_mask = sample_mask.bool()


		return sample, sample_mask


class SpecDataset(torch.utils.data.Dataset):

	def __init__(self, data):
		self.data = data
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):

		sample = self.data[idx]

		return sample, sample

