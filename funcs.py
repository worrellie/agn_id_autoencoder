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

def plot_loss(model_losses, test_name, test=False):

	# plots loss in SCALED SPACE

	train_loss = model_losses["train_total"]
	valid_loss = model_losses["valid_total"]
	train_mse = model_losses["train_mse"]
	valid_mse = model_losses["valid_mse"]
	train_kl = model_losses["train_kl_raw"]
	valid_kl = model_losses["valid_kl_raw"]

	epochs = range(1, len(train_loss) + 1)

	plt.style.use("fivethirtyeight")
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

	ax1.plot(epochs, train_loss, label="Train Total", alpha=0.8)
	ax1.plot(epochs, valid_loss, label="Valid Total", alpha=0.8, linestyle="--")
	ax1.set_title(f"Total Loss (Beta={model_losses['beta']})")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss Value")
	ax1.legend()

	ax2.plot(epochs, train_mse, label="Train MSE", color="tab:blue")
	ax2.plot(epochs, valid_mse, label="Valid MSE", color="tab:blue", linestyle="--")

	if any(k > 0 for k in train_kl):
		ax2_kl = ax2.twinx()
		ax2_kl.plot(
			epochs, train_kl, label="Train KL (raw)", color="tab:red", alpha=0.6
		)
		ax2_kl.plot(
			epochs,
			valid_kl,
			label="Valid KL (raw)",
			color="tab:red",
			alpha=0.6,
			linestyle="--",
		)
		ax2_kl.set_ylabel("KL Divergence", color="tab:red")
		ax2_kl.tick_params(axis="y", labelcolor="tab:red")

		lines, labels = ax2.get_legend_handles_labels()
		lines2, labels2 = ax2_kl.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc="upper right")
	else:
		ax2.legend()

	ax2.set_title("Reconstruction (MSE) vs Regularization (KL)")
	ax2.set_xlabel("Epochs")
	ax2.set_ylabel("MSE Loss", color="tab:blue")
	ax2.tick_params(axis="y", labelcolor="tab:blue")

	plt.tight_layout()
	if not test:
		pth_fig = path.Path(test_name, f"{test_name}_loss.png")
		pth_obj = path.Path(test_name, f"{test_name}_loss.pkl")
		plt.savefig(pth_fig)
		with open(pth_obj, "wb") as o:
			pkl.dump(fig, o)
	else:
		plt.show()

# def _get_example_specs(loader, model):

# 	device = next(model.parameters()).device

# 	# function to get the key spectra to compare

# 	# temp loader to not shuffle data (so can get the right pairs)
# 	temp_loader = torch.utils.data.DataLoader(
# 		loader.dataset,
# 		batch_size=loader.batch_size,
# 		shuffle=False,
# 	)

# 	train_mean = temp_loader.dataset.mean
# 	train_std = temp_loader.dataset.std
# 	# train_mean = None
# 	# train_std = None
# 	normalize = model.normalize
# 	flux_type = model.flux_type

# 	# logger.info('predicting...')
# 	losses_scaled_space = []
# 	losses_unscaled_space = []
# 	model.eval()
# 	with torch.no_grad():
# 		for x, x_mask in temp_loader:

# 			x_unscaled = x * x_mask

# 			if normalize:
# 				x = (x - train_mean) / train_std  # normalize data
# 				x = x * x_mask  # to ensure instrument gap has 0 flux

# 			x = x.to(device)
# 			x_mask = x_mask.to(device)
# 			x_unscaled = x_unscaled.to(device)

# 			x_hat, mu, logvar = model(x)

# 			mses_scaled_space = loss_calc_per_spec(x_hat, x, x_mask,)
# 			mses_scaled_space = mses_scaled_space.cpu().tolist()
# 			losses_scaled_space.extend(mses_scaled_space)
			
# 			device_std = torch.tensor(train_std, device=device)
# 			device_mean = torch.tensor(train_mean, device=device)

# 			x_hat_unscaled = (x_hat * device_std) + device_mean
# 			x_hat_unscaled = x_hat_unscaled

# 			if flux_type == "log_scale_flux":
# 				x_hat_unscaled = torch.sign(x_hat_unscaled) * torch.expm1(torch.abs(x_hat_unscaled))
# 				x_unscaled = torch.sign(x_unscaled) * torch.expm1(torch.abs(x_unscaled))
# 			elif flux_type=="normalized_flux_cont":
# 				# not yet implemented
# 				x_hat_unscaled = x_hat_unscaled
# 				x_unscaled = x_unscaled
# 			elif flux_type == "normalized_flux_med":
# 				# not yet implemented
# 				x_hat_unscaled = x_hat_unscaled
# 				x_unscaled = x_unscaled

# 			x_hat_unscaled = x_hat_unscaled * x_mask
# 			x_unscaled = x_unscaled * x_mask

# 			mses_unscaled_space = loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask,)
# 			mses_unscaled_space = mses_unscaled_space.cpu().tolist()
# 			losses_unscaled_space.extend(mses_unscaled_space)

# 	to_find_scaled_space = {
# 		"min": np.min(losses_scaled_space),
# 		"max": np.max(losses_scaled_space),
# 		"25th": np.percentile(losses_scaled_space, 25),
# 		"mean": np.mean(losses_scaled_space),
# 		"75th": np.percentile(losses_scaled_space, 75),
# 	}

# 	to_find_unscaled_space = {
# 		"min": np.min(losses_unscaled_space),
# 		"max": np.max(losses_unscaled_space),
# 		"25th": np.percentile(losses_unscaled_space, 25),
# 		"mean": np.mean(losses_unscaled_space),
# 		"75th": np.percentile(losses_unscaled_space, 75),
# 	}

# 	logger.info("getting min, max, mean and quartiles of losses...")

# 	idxs_scaled = []
# 	labels_scaled = []
# 	for key, value in to_find_scaled_space.items():
# 		idx = (np.abs(np.array(losses_scaled_space) - value)).argmin()
# 		idxs_scaled.append(idx)
# 		labels_scaled.append(key)

# 	source_subset_scaled = Subset(loader.dataset, idxs_scaled)
# 	subset_loader_scaled = torch.utils.data.DataLoader(
# 		source_subset_scaled, batch_size=1, shuffle=False
# 	)
# 	subset_loader_scaled.dataset.labels = labels_scaled

# 	idxs_unscaled = []
# 	labels_unscaled = []
# 	for key, value in to_find_unscaled_space.items():
# 		idx = (np.abs(np.array(losses_unscaled_space) - value)).argmin()
# 		idxs_unscaled.append(idx)
# 		labels_unscaled.append(key)

# 	source_subset_unscaled = Subset(loader.dataset, idxs_unscaled)
# 	subset_loader_unscaled = torch.utils.data.DataLoader(
# 		source_subset_unscaled, batch_size=1, shuffle=False
# 	)
# 	subset_loader_unscaled.dataset.labels = labels_unscaled

# 	return subset_loader_scaled, subset_loader_unscaled

# def _predict_examples(subset_loader_scaled, subset_loader_unscaled, model,):

# 	device = next(model.parameters()).device

# 	train_mean = subset_loader_scaled.dataset.dataset.mean
# 	train_std = subset_loader_scaled.dataset.dataset.std
# 	normalize = model.normalize
# 	flux_type = model.flux_type
	
# 	# We will build output templates for both spaces dynamically
# 	spaces = {
# 		"scaled": {
# 			"loader": subset_loader_scaled,
# 			"labels": getattr(subset_loader_scaled.dataset, 'labels', [])
# 			},
#         "unscaled": {
#             "loader": subset_loader_unscaled,
#             "labels": getattr(subset_loader_unscaled.dataset, 'labels', [])
#         }
#     }

# 	results = {}

# 	model.eval()

# 	for space_key, config in spaces.items():
# 		output = {
# 			"recon": [],
# 			"original": [],
# 			"mask": [],
# 			"loss": [],
# 			"label": config["labels"],
# 			"mean": train_mean,
# 			"std": train_std,
# 		}

# 		logger.info("predicting min, max, mean and quartiles...")

# 		with torch.no_grad():
# 			for x, x_mask in config["loader"]:
# 				x_ground_truth = x * x_mask
# 				if normalize:
# 					x = (x - train_mean) / train_std  # normalize data
# 					x = x * x_mask  # to ensure instrument gap has 0 flux

# 				x = x.to(device)
# 				x_mask = x_mask.to(device)
# 				x_ground_truth = x_ground_truth.to(device)

# 				x_hat, mu, logvar = model(x)

# 				if space_key == "scaled":
#                     # Scaled Space targets the raw neural network output values
#                     mses = loss_calc_per_spec(x_hat, x, x_mask)
                    
#                     output["recon"].extend(x_hat.cpu().tolist())
#                     output["original"].extend(x.cpu().tolist())
# 				else:
#                     # Unscaled Space reverses normalization and log transformations
#                     device_std = torch.tensor(train_std, device=device)
#                     device_mean = torch.tensor(train_mean, device=device)
                    
#                     x_hat_unscaled = (x_hat * device_std) + device_mean
                    
#                     if flux_type == "log_scale_flux":
#                         x_hat_unscaled = torch.sign(x_hat_unscaled) * torch.expm1(torch.abs(x_hat_unscaled))
#                         x_ground_truth = torch.sign(x_ground_truth) * torch.expm1(torch.abs(x_ground_truth))
                    
#                     x_hat_unscaled = x_hat_unscaled * x_mask
#                     x_ground_truth = x_ground_truth * x_mask
                    
#                     mses = loss_calc_per_spec(x_hat_unscaled, x_ground_truth, x_mask)
                    
#                     output["recon"].extend(x_hat_unscaled.cpu().tolist())
#                     output["original"].extend(x_ground_truth.cpu().tolist())

# 				output["mask"].extend(x_mask.cpu().tolist())
# 				output["loss"].extend(mses.cpu().tolist())

# 			results[space_key] = output


# 	return results["scaled"], results["unscaled"]

# def _plot_example_specs(output,	l,):

# 	fig = plt.figure(figsize=(20, 10))

# 	# We use 6 columns to allow 3-over-2 centering
# 	# 4 rows: [Fit 1, Res 1, Fit 2, Res 2] - we handle vertical pairs manually
# 	gs = fig.add_gridspec(4, 6, height_ratios=[3, 1, 3, 1], hspace=0.4, wspace=0.4)

# 	# --- TOP ROW (3 Plots) ---
# 	# Column spans: 0-2, 2-4, 4-6
# 	for i in range(3):
# 		# Fit at row 0, Res at row 1
# 		ax_fit = fig.add_subplot(gs[0, 2 * i : 2 * i + 2])
# 		ax_res = fig.add_subplot(gs[1, 2 * i : 2 * i + 2], sharex=ax_fit)

# 		_draw_spec_pair(ax_fit, ax_res, output, i, l)

# 	# --- BOTTOM ROW (2 Plots) ---
# 	# Column spans: 0-3, 3-6 (Centering them)
# 	for i in range(2):
# 		idx = i + 3  # Accessing samples 4 and 5 in your output
# 		# Fit at row 2, Res at row 3
# 		ax_fit = fig.add_subplot(gs[2, 3 * i : 3 * i + 3])
# 		ax_res = fig.add_subplot(gs[3, 3 * i : 3 * i + 3], sharex=ax_fit)

# 		_draw_spec_pair(ax_fit, ax_res, output, idx, l)

# 	plt.tight_layout()
# 	# plt.show()

# 	return fig

# def _draw_spec_pair(ax_fit,	ax_res,	output,	i,	l,):

# 	mean = output["mean"]
# 	std = output["std"]

# 	recon = (np.array(output["recon"][i]) * std) + mean
# 	og = (np.array(output["original"][i]) * std) + mean

# 	mask = np.array(output["mask"][i])

# 	resid = og - recon

# 	recon[mask == 0] = np.nan
# 	og[mask == 0] = np.nan
# 	resid[mask == 0] = np.nan

# 	# Fit Panel
# 	ax_fit.step(
# 		l,
# 		og,
# 		color="black",
# 		linewidth=2,
# 		alpha=0.7,
# 		where="mid",
# 		label="Original Spectrum",
# 	)
# 	ax_fit.step(l, recon, color="red", linewidth=1, where="mid", label="Reconstructed")
# 	ax_fit.set_title(f"{output['label'][i]}, loss: {output['loss'][i]:.5f}")

# 	# Residual Panel
# 	ax_res.scatter(l, resid, color="gray")  # residuals of standardized data
# 	ax_res.axhline(0, color="black", lw=0.8, ls=":")

# def plot_examples_old(loader, model, test_params, test=False):

# 	l = loader.dataset.l

# 	subset_loader = _get_example_specs(loader, model)

# 	output = _predict_examples(subset_loader, model)

# 	fig = _plot_example_specs(
# 		output,
# 		l,
# 	)

# 	fig.suptitle(
# 		f"latent: {test_params['latent_size']}, {test_params['activation_function']}, epochs: {test_params['max_epochs']}"
# 	)

# 	plt.tight_layout()
# 	if not test:
# 		pth_fig = path.Path(test_params["test_name"], f"{test_params['test_name']}.png")
# 		pth_obj = path.Path(test_params["test_name"], f"{test_params['test_name']}.pkl")
# 		plt.savefig(pth_fig)
# 		with open(pth_obj, "wb") as o:
# 			pkl.dump(fig, o)
# 	else:
# 		plt.show()

def plot_examples(loader, model, test_params, test=False):

	device = next(model.parameters()).device

	l = loader.dataset.l
	train_mean = loader.dataset.mean
	train_std = loader.dataset.std
	normalize = model.normalize
	flux_type = model.flux_type

	# temp loader of training data to have *no shuffling* in order to get matching pairs
	temp_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False)

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
			x_unscaled = x_unscaled * x_mask
			x_unscaled = x_unscaled.to(device)

			losses_of_unscaled = loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask)

			for i in range(x.shape[0]):
				outputs.append({
					"mask":              x_mask[i].cpu().numpy().astype(bool),
					"original_scaled":   x[i].cpu().numpy(),
					"recon_scaled":      x_hat[i].cpu().numpy(),
					"loss_scaled":       losses_of_scaled[i].item(),
					"original_unscaled": x_unscaled[i].cpu().numpy(),
					"recon_unscaled":    x_hat_unscaled[i].cpu().numpy(),
					"loss_unscaled":     losses_of_unscaled[i].item(),
				})

	all_losses = np.array([o["loss_scaled"] for o in outputs])
	# get percentiles for examples of wht different loss scores look like
	targets = {
	"min":  np.min(all_losses),
	"25th": np.percentile(all_losses, 25),
	"mean": np.mean(all_losses),
	"75th": np.percentile(all_losses, 75),
	"max":  np.max(all_losses),
	}

	examples = []
	for label, target in targets.items():
		example = outputs[int(np.argmin(np.abs(all_losses - target)))]
		example["label"] = label
		examples.append(example)
	
	# plotting
	for space in ("scaled", "unscaled"):
		fig, axes = plt.subplots(5, 2, figsize=(16, 20))
		fig.suptitle(
			f"{space.upper()} — latent: {test_params['latent_size']}, "
			f"{test_params['activation_function']}, epochs: {test_params['max_epochs']}"
		)
		for ax_row, ex in zip(axes, examples):
			ax_fit, ax_res = ax_row
			og    = ex[f"original_{space}"].copy()
			recon = ex[f"recon_{space}"].copy()
			mask  = ex["mask"]
			og[~mask]    = np.nan
			recon[~mask] = np.nan
			resid = og - recon

			ax_fit.step(l, og,    color="black", lw=1.5, alpha=0.7, where="mid", label="Original")
			ax_fit.step(l, recon, color="red",   lw=1.0,             where="mid", label="Reconstructed")
			ax_fit.set_title(f"{ex['label']}, loss: {ex[f'loss_{space}']:.5f}")
			ax_fit.legend(fontsize=8)
			ax_res.scatter(l, resid, color="gray", s=2)
			ax_res.axhline(0, color="black", lw=0.8, ls=":")
			ax_res.set_ylabel("Residual")

		plt.tight_layout()
		if not test:
			pth = path.Path(test_params["test_name"], f"{test_params['test_name']}_{space}.png")
			plt.savefig(pth)
			plt.close(fig)
		else:
			plt.show()


def save_test_params(test_dict, test_name, test=False):

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
