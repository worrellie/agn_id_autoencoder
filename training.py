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
import warnings
import funcs
import json

import logging

logger = logging.getLogger(__name__)


class Trainer:
	def __init__(
		self, device, test_name, model, optimizer, early_stopping, beta, use_autocast=False, test=False
	):

		self.device = device

		self.test_name = test_name
		self.model = model
		self.optimizer = optimizer
		self.early_stopping = early_stopping
		self.beta = beta
		if self.beta == 0 and self.model.type == "vae":
			warnings.warn("Your beta value for a VAE is 0")

		self.use_autocast = use_autocast
		print(f'Using autocast: {use_autocast}')

		self.test = test

		try:
			import intel_extension_for_pytorch as ipex
			self.use_ipex = True
		except ImportError:
			self.use_ipex = False

		"""
		note: dtype float16 is only for GPU. CPU requires bfloat16
		"""
		if self.device.type == "cpu":
			self.autocast_type = torch.bfloat16
			self.grad_scaler = None
		else:
			self.autocast_type = torch.float16
			# note: GradScaler is only for GPU!
			# self.grad_scaler = torch.GradScaler() # to scale gradients so if they are small in float16
			# they dont become zero
			self.grad_scaler = torch.amp.GradScaler("cuda")

	def get_autocast_context(self):
		"""Helper to return the correct autocast context manager"""
		if self.use_ipex and self.device.type == "cpu":
			import intel_extension_for_pytorch as ipex

			return ipex.autocast(device_type="cpu", dtype=torch.bfloat16)
		else:
			# Fallback to standard torch autocast (handles CUDA or standard CPU)
			return torch.amp.autocast(
				device_type=self.device.type, dtype=self.autocast_type
			)

	def train_ae(self, epochs, train_loader, valid_loader=None, verbose=False):

		train_mean = train_loader.dataset.mean
		train_std = train_loader.dataset.std

		normalize = self.model.normalize
		flux_type = self.model.flux_type
		self.model.mean = train_mean
		self.model.std = train_std

		# normalize = False
		clip = False

		# if train_mean is not None and train_std is not None:
		# 	normalize = True

		# # FOR TESTING
		# normalize = False
		# clip = False
		# #############

		if clip:
			print(f"Applying clipping to input data (clipped beyond -5)")
		if normalize:
			print(f"Applying Z-score normalization to input data")
		if not normalize and not clip:
			print(f"No clipping or normalization applied")

		self.model.to(self.device)

		train_losses = []
		train_mses = []
		train_kls = []
		valid_losses = []
		valid_mses = []
		valid_kls = []

		print(f'norm: {normalize}')
		print(f'clip: {clip}')

		logger.info("training model...")

		for epoch in range(epochs):
			self.model.train()
			train_loss = 0
			train_mse = 0
			train_kl = 0
			valid_loss = 0
			valid_mse = 0
			valid_kl = 0

			processed_samples = 0

			first_param = next(self.model.parameters())
			print(first_param.shape)
			logger.info(f"epoch {epoch} first param mean: {first_param.data.mean():.6f}")
			logger.info(f"epoch {epoch} first x_hat mean: check below")

			for x, x_mask in train_loader:

				# for understanding exploding gradient problem
				# check min and max incoming x values
				all_vals = []
				all_vals.append(x[x_mask])
				if len(all_vals) > 20:
					break
				all_vals = torch.cat(all_vals)
				print(f"Raw data: min={all_vals.min():.3f}, max={all_vals.max():.3f}")
				print(f"          mean={all_vals.mean():.3f}, std={all_vals.std():.3f}")
				print(f"          >10: {(all_vals.abs() > 10).sum()}, >100: {(all_vals.abs() > 100).sum()}")

				stded = (all_vals - train_mean)/ train_std
				print(f"After standardising: min={stded.min():.3f}, max={stded.max():.3f}")
				print(f"                       std={stded.std():.3f}")
				#############################################################################################

				if normalize:
					x = (x - train_mean) / train_std  # normalize data
					x = x * x_mask  # re-set 'gaps'/masked regions as zero
				if clip:
					x = torch.clamp(x, min = -5.0)
					x = x * x_mask

				x = x.to(self.device)
				x_mask = x_mask.to(self.device)

				"""
				consider the case for autocasting: float32 takes up a lot of memory but is very precise.
				autocast can switch to float16 (50% less memory) for speed and then
				back to float32 when precision is critical.
				warning: very small gradients can turn into zero in float16.
				to prevent this, can use GradScaler to scale up weights for calculations
				then shrink them back down before optmizer updates the weights
				"""

				self.optimizer.zero_grad()

				if self.use_autocast:

					print('autocast in action')

					with self.get_autocast_context():

						x_hat, mu, logvar = self.model(x)  # batch prediction. note: only VAE will output non-None mu/var
						logger.info(f"train x_hat mean: {x_hat.mean().item():.6f}, x mean: {x.mean().item():.6f}")

						# stats for *batch*
						mse, kl, loss = funcs._loss_calc_batch(x_hat, x, x_mask, mu=mu, logvar=logvar, beta=self.beta)  # 'mean' gives loss per sample for batch
						print(loss)
						print(loss.data.mean())

					if self.grad_scaler is not None:
						self.grad_scaler.scale(loss).backward()  # call backward on scaled loss to create scaled
						# scaled gradients
						self.grad_scaler.step(self.optimizer)  # scaler.step unscales gradients. then if theyre not
						# inf or nan, step is called. otherwise step is skipped
						self.grad_scaler.update()  # update scales for next iteration
					else:
						loss.backward()
						self.optimizer.step()

				else:

					x_hat, mu, logvar = self.model(x)  # batch prediction. note: only VAE will output non-None mu/var
					logger.info(f"train x_hat mean: {x_hat.mean().item():.6f}, x mean: {x.mean().item():.6f}")

					# stats for *batch*
					mse, kl, loss = funcs._loss_calc_batch(x_hat, x, x_mask, mu=mu, logvar=logvar, beta=self.beta)  # 'mean' gives loss per sample for batch
					print(loss)
					print(loss.data.mean())

					loss.backward()

					self.optimizer.step()

				# note: .item() in pytorch gives UNSCALED loss
				train_mse += mse.item() * x.size(0)  # reconstruction loss per sample
				# mse.item is batch mean therefore need to multiply by batch size
				# and later divide by total number of samples to get epoch avg
				train_kl += kl.item() * x.size(0)  # kl divergence
				# train_w_kls += w_kl.item() / x.size(0) # weighted kl divergence

				train_loss += loss.item() * x.size(0)  # total loss

				# in case drop_last is True, divide by number used, rather than dataset size
				processed_samples += x.size(0)

			epoch_avg_mse = train_mse / processed_samples
			train_mses.append(epoch_avg_mse)
			epoch_avg_kl = train_kl / processed_samples
			train_kls.append(epoch_avg_kl)

			epoch_avg_loss = train_loss / processed_samples  # average loss per sample
			train_losses.append(epoch_avg_loss)  # losses for each epoch

			logger.info("-------------------------------------------")
			logger.info(
				f"training: epoch {epoch + 1}/{epochs},\ntotal loss: {epoch_avg_loss:.10f},\nmse: {epoch_avg_mse:.10f},\nkl: {epoch_avg_kl:e}"
			)

			if valid_loader is not None:
				# dont update weights/ train
				self.model.eval()
				with torch.no_grad():
					first_param = next(self.model.parameters())
					logger.info(
						f"epoch {epoch} first param mean: {first_param.data.mean():.6f}"
					)
					logger.info(f"epoch {epoch} first x_hat mean: check below")

					processed_samples_valid = 0

					# print('normalize for validation')

					for x, x_mask in valid_loader:
						if normalize:
							x = (x - train_mean) / train_std  # normalize data
							x = x * x_mask  # re-set 'gaps'/masked regions as zer

						x = x.to(self.device)
						x_mask = x_mask.to(self.device)

						x_hat, mu, logvar = self.model(x)

						logger.info(
							f"valid x_hat mean: {x_hat.mean().item():.6f}, x mean: {x.mean().item():.6f}"
						)

						mse, kl, loss = funcs._loss_calc_batch(x_hat, x, x_mask, mu=mu, logvar=logvar, beta=self.beta)  # 'mean' gives loss per sample for batch
						print(loss)
						print(loss.data.mean())

						# print(x.size(0))

						valid_mse += mse.item() * x.size(0)  # reconstruction loss
						valid_kl += kl.item() * x.size(0)  # kl divergence
						# valid_w_kls += w_kl.item() / x.size(0) # weighted kl divergence

						valid_loss += loss.item() * x.size(0)

						# valid_samples = len(valid_loader.dataset)

						processed_samples_valid += x.size(0)

					epoch_avg_valid_mse = valid_mse / processed_samples_valid
					valid_mses.append(epoch_avg_valid_mse)
					epoch_avg_valid_kl = valid_kl / processed_samples_valid
					valid_kls.append(epoch_avg_valid_kl)

					epoch_avg_valid_loss = valid_loss / processed_samples_valid
					valid_losses.append(epoch_avg_valid_loss)

				logger.info(
					f"valid: epoch {epoch + 1}/{epochs},\ntotal loss: {epoch_avg_valid_loss:.10f},\nmse: {epoch_avg_valid_mse:.10f},\nkl: {epoch_avg_valid_kl:e}"
				)

				if self.early_stopping is not None:
					self.early_stopping.check_early_stop(
						epoch_avg_valid_loss, self.model, epoch
					)

					if self.early_stopping.stop_training:
						logger.info("---------------------------------")
						logger.info(f"Early Stopping: epoch {epoch}")
						logger.info("---------------------------------")
						break

		logger.info("training finished")
		# when at end of training, save (if not a test)
		if not self.test:
			save_path_dict = path.Path(
				self.test_name, f"{self.test_name}_state_dict.pt"
			)  # overwrite is default
			torch.save(self.model.state_dict(), save_path_dict)
			save_path_model = path.Path(self.test_name, f"{self.test_name}_model.pt")
			torch.save(self.model, save_path_model)

		model_losses = {
			"beta": self.beta,
			"train_total": train_losses,
			"train_mse": train_mses,
			"train_kl_raw": train_kls,
			"valid_total": valid_losses,
			"valid_mse": valid_mses,
			"valid_kl_raw": valid_kls,
		}

		# save model losses: (if not testing)
		if not self.test:
			loss_pth = path.Path(self.test_name, f"{self.test_name}_losses.json")
			with open(loss_pth, "w") as p:
				json.dump(model_losses, p)

		return self.model, model_losses


class CustomEarlyStopping:
	def __init__(self, test_name, patience=5, delta=0, test=False, verbose=False):

		self.patience = patience
		self.delta = delta
		self.verbose = verbose
		self.best_loss = None
		self.no_improve_count = 0
		self.stop_training = False
		self.test = test
		self.test_name = test_name

	def save_model(self, model, test_name, epoch):

		save_path_dict = path.Path(
			test_name, f"{test_name}_{self.patience}_{self.delta}_state_dict.pt"
		)  # overwrite is default
		torch.save(model.state_dict(), save_path_dict)
		save_path_model = path.Path(
			test_name, f"{test_name}_{self.patience}_{self.delta}_model.pt"
		)  # overwrite is default
		torch.save(model, save_path_model)

	def check_early_stop(self, validation_loss, model, epoch):

		if self.best_loss is None or validation_loss < self.best_loss - self.delta:
			# if starting or if new validation loss is better than current best loss
			# set best loss as new validaton loss and reset count of no improvement
			# and save best model
			self.best_loss = validation_loss
			if not self.test:
				self.save_model(model, self.test_name, epoch)
			self.no_improve_count = 0
		else:
			# if new validation loss is not better, increase count of no improvement
			self.no_improve_count += 1
			if self.no_improve_count >= self.patience:
				# no improve count reaches patience, stop
				self.stop_training = True
				if self.verbose:
					logger.info("Stopping Early")
