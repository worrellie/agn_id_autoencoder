import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pathlib as path
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import pickle as pkl
from torch.utils.data import Subset
import json

import wandb

import warnings


# from ignite.engine import Engine, Events
# from ignite.handlers import ModelCheckpoint

import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_size_mb(model):
	# Calculate parameters (weights that are trained)
	param_size = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()

	# Calculate buffers (fixed tensors like running means)
	buffer_size = 0
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()

	total_size_mb = (param_size + buffer_size) / 1024**2

	logger.info(f"model size: {total_size_mb}")

	return total_size_mb

def _loss_calc_batch(x_hat,	x,	x_mask,	mu=None, logvar=None, beta=0,):
	"""
	function to get average loss of batch
	"""

	batch_size = x_hat.shape[0]
	n_unmasked_pixels = x_mask.sum(dim=1)

	# pixel-wise
	sq_err_per_element = (x_hat - x) ** 2

	# apply masks
	masked_sq_err = sq_err_per_element * x_mask

	# mse per spec
	masked_mse_per_sample = masked_sq_err.sum(dim=1) / n_unmasked_pixels

	# mean mse for batch
	mean_masked_mse_for_batch = masked_mse_per_sample.sum() / batch_size
	if mu is not None and logvar is not None:  # (if is VAE)
		# kl divs in latent space (one for each dim of latent space):
		kl_divs = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
		mean_kl_div_for_batch = kl_divs.sum() / batch_size
	else:
		mean_kl_div_for_batch = torch.tensor(0.0).to(x.device)

	recon_loss = mean_masked_mse_for_batch
	kl_loss = mean_kl_div_for_batch

	total_loss = recon_loss + (beta * kl_loss)

	return recon_loss, kl_loss, total_loss

def _rel_mse_calc_batch(x_hat,	x,	x_mask,	):
	"""
	function to get average loss of batch
	"""

	batch_size = x_hat.shape[0]
	n_unmasked_pixels = x_mask.sum(dim=1)

	# pixel-wise
	sq_err_per_element = (x_hat - x) ** 2

	rel_sq_err_per_element = sq_err_per_element / (x.abs() + 1e-10) # epsilon to not fail for 0 value pixels

	# apply masks
	masked_sq_err = rel_sq_err_per_element * x_mask

	# rel mse per spec
	masked_rel_mse_per_sample = masked_sq_err.sum(dim=1) / n_unmasked_pixels

	# rel mean mse for batch
	rel_mean_masked_mse_for_batch = masked_rel_mse_per_sample.sum() / batch_size

	return rel_mean_masked_mse_for_batch

def loss_calc_per_spec(x_hat, x, x_mask,):
	"""
	function to get MSE of each spectrum in batch
	return list of MSEs that is same length as number of spec n batch
	"""

	batch_size = x_hat.shape[0]

	n_unmasked_pixels = x_mask.sum(dim=1)

	# pixel-wise
	sq_err_per_element = (x_hat - x) ** 2

	# apply masks
	masked_sq_err = sq_err_per_element * x_mask

	# mse per spec
	recon_loss = masked_sq_err.sum(dim=1) / n_unmasked_pixels

	return recon_loss

def rel_loss_calc_per_spec(x_hat, x, x_mask,):
	"""
	function to get MSE of each spectrum in batch
	return list of MSEs that is same length as number of spec n batch
	"""

	batch_size = x_hat.shape[0]

	n_unmasked_pixels = x_mask.sum(dim=1)

	# pixel-wise
	sq_err_per_element = (x_hat - x) ** 2

	rel_sq_err_per_element = sq_err_per_element / (x.abs() + 1e-10)

	# apply masks
	masked_sq_err = rel_sq_err_per_element * x_mask

	# mse per spec
	recon_loss = masked_sq_err.sum(dim=1) / n_unmasked_pixels

	return recon_loss

def get_predictions(loader, model, test_params, test = False):
	# for getting metrics from a final model

	test_name = test_params["test_name"]

	device = next(model.parameters()).device

	d_split = loader.dataset.split

	l = loader.dataset.l
	train_mean = loader.dataset.mean
	train_std = loader.dataset.std
	normalize = model.normalize
	flux_type = model.flux_type

	# temp loader of training data to have *no shuffling* in order to get matching pairs
	temp_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False)
	## ok so im not sure shuffle true would be an issue in this use?? check...
	## because even with batching it only shuffles at the after going through *all* the samples...
	# so there wouldnt be an issue and it would be faster if batched....

	# get all losses and store (negligible memory usage)	
	outputs = []
	model.eval()
	with torch.no_grad():
		for x, x_mask in temp_loader:
			# z score standardize if normalize is True
			x_unscaled = x * x_mask # store unstandardized x (still log/cont/mean scaled!!)
			x = ((x - train_mean) / train_std) * x_mask if normalize else x * x_mask
			x = x.to(device)
			x_mask = x_mask.to(device)

			x_hat, _, _ = model(x)

			losses_of_scaled = loss_calc_per_spec(x_hat, x, x_mask)

			x_hat_unscaled = (x_hat * train_std + train_mean) if normalize else x_hat
			if flux_type == "log_scale_flux":
				x_hat_unscaled = torch.sign(x_hat_unscaled) * torch.expm1(torch.abs(x_hat_unscaled))
				x_unscaled = torch.sign(x_unscaled.to(device)) * torch.expm1(torch.abs(x_unscaled.to(device)))
			elif flux_type == "normalized_flux_cont":
				pass # not added yet
			elif flux_type == "normalized_flux_med":
				pass # not addded yet
			x_hat_unscaled = x_hat_unscaled * x_mask
			x_unscaled = x_unscaled.to(device) * x_mask
			x_unscaled = x_unscaled

			losses_of_unscaled = loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask)

			rel_losses = rel_loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask)

			for i in range(x.shape[0]): # for each spectrum make a dictionary with the info
				outputs.append({
					"mask": x_mask[i].cpu().numpy().astype(bool).tolist(), # spectrum mask
					"original_scaled": x[i].cpu().numpy().tolist(), # original spectrum in scaled space
					"recon_scaled": x_hat[i].cpu().numpy().tolist(), # reconstructed spectrum in scaled space
					"loss_scaled": losses_of_scaled[i].item(), # MSE of spectrum reconstruction in scaled space
					"original_unscaled": x_unscaled[i].cpu().numpy().tolist(), # original spectrum RAW, UNSCALED space
					"recon_unscaled": x_hat_unscaled[i].cpu().numpy().tolist(), ## reconstructed spectrum RAW, UNSCALED space
					"loss_unscaled": losses_of_unscaled[i].item(), # MSE of spectrum reconstruction in UNSCALED space
					"rel_loss" : rel_losses[i].item(), # relative MSE in UNSCALED space
				})

	# save outputs
	if not test:
		pth = path.Path(test_name, f"{test_name}_{d_split}_outputs.json")
		with open(pth, "w") as p:
			json.dump(outputs, p)

	return outputs

def model_stats(outputs, test_params, best):

	test_name = test_params["test_name"]

	all_losses_scaled = np.array([o["loss_scaled"] for o in outputs])

	all_losses_unscaled = np.array([o["loss_unscaled"] for o in outputs])

	all_rel_losses = np.array([o["rel_loss"] for o in outputs])

	loss_stats = {
		"scaled": {
			"mean":   float(np.mean(all_losses_scaled)),
			"median": float(np.median(all_losses_scaled)),
			"std":    float(np.std(all_losses_scaled)),
			"p95":    float(np.percentile(all_losses_scaled, 95)),
			"max":    float(np.max(all_losses_scaled)),
		},
		"unscaled": {
			"mean":   float(np.mean(all_losses_unscaled)),
			"median": float(np.median(all_losses_unscaled)),
			"std":    float(np.std(all_losses_unscaled)),
			"p95":    float(np.percentile(all_losses_unscaled, 95)),
			"max":    float(np.max(all_losses_unscaled)),
		},
		"rel": {
			"mean":   float(np.mean(all_rel_losses)),
			"median": float(np.median(all_rel_losses)),
			"std":    float(np.std(all_rel_losses)),
			"p95":    float(np.percentile(all_rel_losses, 95)),
			"max":    float(np.max(all_rel_losses)),
		},
	} 
	
	stats_type = "best" if best else "final"
	path_name = path.Path(test_name, f"{test_name}_{stats_type}_model_stats.json")

	with open(path_name, "w") as p:
		json.dump(loss_stats, p, indent=4)	


	return loss_stats

def save_test_params(test_dict, test_params, test=False):

	test_name = test_params["test_name"]

	if test:
		return

	path.Path(test_name).mkdir(
		parents=False, exist_ok=True
	)  # folder should already exist
	path_name = path.Path(test_name, f"{test_name}_params.json")

	with open(path_name, "w") as p:
		json.dump(test_dict, p, indent=4)

def make_test_dir(test_name, test=False):

	if test:
		return

	path.Path(test_name).mkdir(parents=False, exist_ok=False)

def global_stats(loader):

	all_fluxes = []
	for batch_flux, batch_mask in loader:
		mask = batch_flux != 0
		all_fluxes.append(batch_flux[mask])

	combined_fluxes = torch.cat(all_fluxes)

	return combined_fluxes.mean(), combined_fluxes.std()

def _to_physical_space(x_scaled, train_mean, train_std, normalize, flux_type):

	# Step 1: invert z-score
	if normalize:
		x = (x_scaled * train_std) + train_mean
	else:
		x = x_scaled.clone()

	# Step 2: invert flux-space transform
	if flux_type == "log_scale_flux":
		x = torch.sign(x) * torch.expm1(torch.abs(x))
	elif flux_type == "normalized_flux_cont":
		pass # not added yet
	elif flux_type == "normalized_flux_med":
		pass # not addded yet                          # raw flux, z-score inversion is sufficient
	else:
		logger.warning(f"Unknown flux_type '{flux_type}' — no flux-space inversion applied")

	return x

def log_final_stats(losses_per_epoch):

	wandb.log({
		"final/train_loss": losses_per_epoch["train_total"][-1],
		"final/valid_loss": losses_per_epoch["valid_total"][-1],
		#
		"final/best_valid_scaled": min(losses_per_epoch["valid_total"]),
		"final/best_valid_unscaled_mse": min(losses_per_epoch["unscaled_valid_mses"]),  # target for wandb sweep
		"final/best_valid_unscaled_rel_mse": min(losses_per_epoch["unscaled_valid_rel_mses"]),
		#
		"final/best_epoch_scaled": int(np.argmin(losses_per_epoch["valid_total"])),
		"final/train_at_best_scaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["valid_total"]))],
		"final/overfit_gap_scaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["valid_total"]))] - min(losses_per_epoch["valid_total"]),
		#
		"final/best_epoch_unscaled": int(np.argmin(losses_per_epoch["unscaled_valid_mses"])),
		"final/train_at_best_unscaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["unscaled_valid_mses"]))],
		#			
		"final/best_epoch_rel_unscaled": int(np.argmin(losses_per_epoch["unscaled_valid_rel_mses"])),
		"final/train_at_best_rel_unscaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["unscaled_valid_rel_mses"]))],
	})

def log_summary(train_outputs, valid_outputs, test_params, test = False):

	test_name = test_params["test_name"]

	# all_losses_scaled = np.array([o["loss_scaled"] for o in outputs])

	train_all_losses_unscaled = np.array([o["loss_unscaled"] for o in train_outputs])
	train_all_losses_scaled = np.array([o["loss_scaled"] for o in train_outputs])
	train_all_rel_losses = np.array([o["rel_loss"] for o in train_outputs])

	valid_all_losses_unscaled = np.array([o["loss_unscaled"] for o in valid_outputs])
	valid_all_losses_scaled = np.array([o["loss_scaled"] for o in valid_outputs])
	valid_all_rel_losses = np.array([o["rel_loss"] for o in valid_outputs])

	wandb.run.summary["mean_unscaled_mse"] = np.mean(valid_all_losses_unscaled)
	wandb.run.summary["mean_rel_mse"] = np.mean(valid_all_rel_losses) # unscaled i think...
	wandb.run.summary["mean_overfit"] = np.mean(train_all_rel_losses) - np.mean(valid_all_rel_losses)

def get_latent_space(loader, model, test_params, test=False):

	test_name = test_params["test_name"]

	device = next(model.parameters()).device
	d_split = loader.dataset.split

	train_mean = loader.dataset.mean
	train_std = loader.dataset.std
	normalize = model.normalize
	flux_type = model.flux_type

	temp_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False)

	all_latent = []
	all_loss_scaled = []
	all_loss_unscaled = []
	all_rel_loss = []

	model.eval()
	with torch.no_grad():
		for x, x_mask in temp_loader:
			x_unscaled = x * x_mask
			x = ((x - train_mean) / train_std) * x_mask if normalize else x * x_mask
			x = x.to(device)
			x_mask = x_mask.to(device)

			z = model.encode(x)
			all_latent.append(z.cpu().numpy())

			x_hat, _, _ = model(x)

			all_loss_scaled.append(loss_calc_per_spec(x_hat, x, x_mask).cpu().numpy())

			x_hat_unscaled = (x_hat * train_std + train_mean) if normalize else x_hat
			if flux_type == "log_scale_flux":
				x_hat_unscaled = torch.sign(x_hat_unscaled) * torch.expm1(torch.abs(x_hat_unscaled))
				x_unscaled = torch.sign(x_unscaled.to(device)) * torch.expm1(torch.abs(x_unscaled.to(device)))
			x_hat_unscaled = x_hat_unscaled * x_mask
			x_unscaled = x_unscaled.to(device) * x_mask

			all_loss_unscaled.append(loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask).cpu().numpy())
			all_rel_loss.append(rel_loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask).cpu().numpy())

	latent = np.concatenate(all_latent, axis=0)
	loss_scaled = np.concatenate(all_loss_scaled)
	loss_unscaled = np.concatenate(all_loss_unscaled)
	rel_loss = np.concatenate(all_rel_loss)

	redshift = temp_loader.dataset._get_redshift()
	snr = temp_loader.dataset._get_snr()

	latent_data = {
		"latent":        latent,
		"loss_scaled":   loss_scaled,
		"loss_unscaled": loss_unscaled,
		"rel_loss":      rel_loss,
		"redshift":      redshift,
		"snr":           snr,
	}

	if not test:
		save_arrays = {k: v for k, v in latent_data.items() if v is not None}
		pth = path.Path(test_name, f"{test_name}_{d_split}_latent.npz")
		np.savez(pth, **save_arrays)

	return latent_data

	