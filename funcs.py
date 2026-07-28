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
import os
import h5py
import gc
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
	n_unmasked_pixels = x_mask.sum(dim=1).clamp(min=1) # clamp to protect against zero division

	# pixel-wise
	sq_err_per_element = (x_hat - x) ** 2

	# apply masks
	masked_sq_err = torch.where(x_mask, sq_err_per_element, torch.zeros_like(sq_err_per_element))

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

# def _rel_mse_calc_batch(x_hat,	x,	x_mask,	):
# 	"""
# 	function to get average loss of batch
# 	"""

# 	batch_size = x_hat.shape[0]
# 	n_unmasked_pixels = x_mask.sum(dim=1).clamp(min=1) # clamp to protect against zero division

# 	# pixel-wise
# 	sq_err_per_element = (x_hat - x) ** 2

# 	rel_sq_err_per_element = sq_err_per_element / (x.abs() + 1e-10) # epsilon to not fail for 0 value pixels

# 	# apply masks
# 	masked_sq_err = torch.where(x_mask, sq_err_per_element, torch.zeros_like(sq_err_per_element))

# 	# rel mse per spec
# 	masked_rel_mse_per_sample = masked_sq_err.sum(dim=1) / n_unmasked_pixels

# 	# rel mean mse for batch
# 	rel_mean_masked_mse_for_batch = masked_rel_mse_per_sample.sum() / batch_size

# 	return rel_mean_masked_mse_for_batch

def loss_calc_per_spec(x_hat, x, x_mask,):
	"""
	function to get MSE of each spectrum in batch
	return list of MSEs that is same length as number of spec n batch
	"""

	batch_size = x_hat.shape[0]

	n_unmasked_pixels = x_mask.sum(dim=1).clamp(min=1) # clamp to protect against zero division

	# pixel-wise
	sq_err_per_element = (x_hat - x) ** 2

	# apply masks
	masked_sq_err = torch.where(x_mask, sq_err_per_element, torch.zeros_like(sq_err_per_element))

	# mse per spec
	recon_loss = masked_sq_err.sum(dim=1) / n_unmasked_pixels

	return recon_loss

# def rel_loss_calc_per_spec(x_hat, x, x_mask,):
# 	"""
# 	function to get MSE of each spectrum in batch
# 	return list of MSEs that is same length as number of spec n batch
# 	"""

# 	batch_size = x_hat.shape[0]

# 	n_unmasked_pixels = x_mask.sum(dim=1).clamp(min=1) # clamp to protect against zero division


# 	# pixel-wise
# 	sq_err_per_element = (x_hat - x) ** 2

# 	rel_sq_err_per_element = sq_err_per_element / (x.abs() + 1e-10)

# 	# apply masks
# 	masked_sq_err = torch.where(x_mask, sq_err_per_element, torch.zeros_like(sq_err_per_element))

# 	# mse per spec
# 	recon_loss = masked_sq_err.sum(dim=1) / n_unmasked_pixels

# 	return recon_loss

# def get_predictions(loader, model, test_params, test = False):
# 	# for getting metrics from a final model

# 	test_name = test_params["test_name"]

# 	device = next(model.parameters()).device

# 	ds = loader.dataset
# 	d_split = ds.split

# 	l = ds.l
# 	train_mean = ds.mean
# 	train_std = ds.std
# 	standardize = ds.standardize
# 	flux_type = ds.flux_type

# 	# temp loader of training data to have *no shuffling* in order to get matching pairs
# 	temp_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False)
# 	## ok so im not sure shuffle true would be an issue in this use?? check...
# 	## because even with batching it only shuffles at the after going through *all* the samples...
# 	# so there wouldnt be an issue and it would be faster if batched....

# 	# get all losses and store (negligible memory usage)
# 	ls, lu = [], []
# 	model.eval()
# 	with torch.no_grad():

# 		for x, x_mask in temp_loader:

# 			x = x.to(device)
# 			x_mask = x_mask.to(device)

# 			x_hat, _, _ = model(x)

# 			# losses_of_scaled = loss_calc_per_spec(x_hat, x, x_mask)
# 			ls.append(loss_calc_per_spec(x_hat, x, x_mask).cpu().numpy())

# 			x_unscaled = _to_physical_space(x, train_mean, train_std, standardize, flux_type, mask = x_mask)
# 			x_hat_unscaled = _to_physical_space(x_hat, train_mean, train_std, standardize, flux_type, mask = x_mask)
# 			lu.append(loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask).cpu().numpy())

# 			# losses_of_unscaled = loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask)

# 		loss_scaled = np.concatenate(ls)
# 		loss_unscaled = np.concatenate(lu)

# 		def _pick_spec(arr):
# 			targets = {"min": arr.min(), "25th": np.percentile(arr, 25), "mean": arr.mean(),
# 				"75th": np.percentile(arr, 75), "max": arr.max()}
# 			return {lbl: int(np.argmin(np.abs(arr - t))) for lbl, t in targets.items()}
		
# 		picked_specs = {"scaled": _pick_spec(loss_scaled), "unscaled": _pick_spec(loss_unscaled)}

# 		examples = {}
# 		for space, sel in picked_specs.items():
# 			rows = []
# 			for lbl, i in sel.items():
# 				x, m = ds[i]
# 				x = x.unsqueeze(0).to(device)
# 				m = m.unsqueeze(0).to(device)
# 				x_hat, _, _ = model(x)
# 				if space == "scaled":
# 					og, rec = x, x_hat
# 				else:
# 					og  = _to_physical_space(x,     train_mean, train_std, standardize, flux_type, mask=m)
# 					rec = _to_physical_space(x_hat, train_mean, train_std, standardize, flux_type, mask=m)
# 				rows.append({
# 					"label": lbl,
# 					"loss":  float((loss_scaled if space == "scaled" else loss_unscaled)[i]),
# 					"mask":  m[0].cpu().numpy(),                       # bool array, not a list
# 					"og":    og[0].cpu().numpy().astype(np.float32),   # float32, not Python floats
# 					"recon": rec[0].cpu().numpy().astype(np.float32),
# 				})
# 			examples[space] = rows

# 		out = {"loss_scaled": loss_scaled, "loss_unscaled": loss_unscaled,
# 			"examples": examples, "split": ds.split}

# 		if not test:
# 			# npz, not JSON. float32 arrays instead of a multi-GB string.
# 			np.savez_compressed(
# 				path.Path(test_params["test_name"], f"{test_params['test_name']}_{ds.split}_losses.npz"),
# 				loss_scaled=loss_scaled, loss_unscaled=loss_unscaled,
# 			)

# 		# returns the specs of only the example spectra
# 		return out


# 	# 		for i in range(x.shape[0]): # for each spectrum make a dictionary with the info
# 	# 			outputs.append({
# 	# 				"mask": x_mask[i].cpu().numpy().astype(bool).tolist(), # spectrum mask
# 	# 				"original_scaled": x[i].cpu().numpy().tolist(), # original spectrum in scaled space
# 	# 				"recon_scaled": x_hat[i].cpu().numpy().tolist(), # reconstructed spectrum in scaled space
# 	# 				"loss_scaled": losses_of_scaled[i].item(), # MSE of spectrum reconstruction in scaled space
# 	# 				"original_unscaled": x_unscaled[i].cpu().numpy().tolist(), # original spectrum RAW, UNSCALED space
# 	# 				"recon_unscaled": x_hat_unscaled[i].cpu().numpy().tolist(), ## reconstructed spectrum RAW, UNSCALED space
# 	# 				"loss_unscaled": losses_of_unscaled[i].item(), # MSE of spectrum reconstruction in UNSCALED space
# 	# 				# "rel_loss" : rel_losses[i].item(), # relative MSE in UNSCALED space
# 	# 			})

# 	# # save outputs
# 	# if not test:
# 	# 	pth = path.Path(test_name, f"{test_name}_{d_split}_outputs.json")
# 	# 	with open(pth, "w") as p:
# 	# 		json.dump(outputs, p)

# 	return outputs

def evaluate(loader, model, test_params, want_latent=False, want_examples=False, test=False):
	"""
	ONE forward pass over a split. Replaces get_predictions + get_latent_space,
	which each looped the whole split and both recomputed the same per-spec losses.

	Returns a dict of ARRAYS (never a list of 110k dicts — that was ~32 GB of Python floats):
	  loss_scaled   : (N,) per-spectrum MSE in standardised space
	  loss_unscaled : (N,) per-spectrum MSE in physical space
	  latent        : (N, D)  if want_latent
	  redshift, snr : (N,)    if present in the HDF5
	  examples      : {"scaled": [5 rows], "unscaled": [5 rows]}  if want_examples
	  n_dead, n_eff : latent health   if want_latent

	sweep mode:  want_latent=True,  want_examples=False
	full mode:   want_latent=True,  want_examples=True
	"""
	device = next(model.parameters()).device

	ds          = loader.dataset
	d_split     = ds.split
	train_mean  = ds.mean
	train_std   = ds.std
	standardize = ds.standardize
	flux_type   = ds.flux_type

	# shuffle=False: latent rows must stay aligned with the HDF5 row order,
	# because redshift/snr are read straight from the file by index.
	temp_loader = torch.utils.data.DataLoader(ds, batch_size=loader.batch_size, shuffle=False)

	# ---------- PASS 1: scalars (+ latent if asked) ----------
	ls, lu, lat = [], [], []

	model.eval()
	with torch.no_grad():
		for x, x_mask in temp_loader:
			x      = x.to(device)
			x_mask = x_mask.to(device)

			x_hat, _, _ = model(x)

			ls.append(loss_calc_per_spec(x_hat, x, x_mask).cpu().numpy())

			x_unscaled     = _to_physical_space(x,     train_mean, train_std, standardize, flux_type, mask=x_mask)
			x_hat_unscaled = _to_physical_space(x_hat, train_mean, train_std, standardize, flux_type, mask=x_mask)
			lu.append(loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask).cpu().numpy())

			if want_latent:
				lat.append(model.encode(x).cpu().numpy())

	loss_scaled   = np.concatenate(ls)      # ONCE, outside the loop
	loss_unscaled = np.concatenate(lu)

	out = {
		"split":         d_split,
		"loss_scaled":   loss_scaled,
		"loss_unscaled": loss_unscaled,
		"redshift":      ds._get_redshift(),
		"snr":           ds._get_snr(),
	}

	# ---------- latent health ----------
	if want_latent:
		latent = np.concatenate(lat, axis=0)
		out["latent"] = latent

		# a unit with zero variance across the split carries no information:
		# your EFFECTIVE latent size is smaller than latent_size claims.
		# Should be 0 now the bottleneck is linear — log it to PROVE that.
		unit_std = latent.std(axis=0)
		out["n_dead"] = int((unit_std < 1e-8).sum())
		out["n_eff"]  = int(latent.shape[1] - out["n_dead"])
		if out["n_dead"]:
			logger.warning(f"[{d_split}] {out['n_dead']} dead latent units — "
			               f"effective latent size {out['n_eff']}, not {latent.shape[1]}")

	# ---------- PASS 2: re-run ONLY the 10 spectra we actually plot ----------
	if want_examples:
		def _pick(arr):
			targets = {"min":  arr.min(),
			           "25th": np.percentile(arr, 25),
			           "mean": arr.mean(),
			           "75th": np.percentile(arr, 75),
			           "max":  arr.max()}
			return {lbl: int(np.argmin(np.abs(arr - t))) for lbl, t in targets.items()}

		picks = {"scaled": _pick(loss_scaled), "unscaled": _pick(loss_unscaled)}

		examples = {}
		with torch.no_grad():
			for space, sel in picks.items():
				rows = []
				for lbl, i in sel.items():
					x, m = ds[i]
					x = x.unsqueeze(0).to(device)
					m = m.unsqueeze(0).to(device)
					x_hat, _, _ = model(x)

					if space == "scaled":
						og, rec = x, x_hat
					else:
						og  = _to_physical_space(x,     train_mean, train_std, standardize, flux_type, mask=m)
						rec = _to_physical_space(x_hat, train_mean, train_std, standardize, flux_type, mask=m)

					rows.append({
						"label": lbl,
						"index": i,
						"loss":  float((loss_scaled if space == "scaled" else loss_unscaled)[i]),
						"mask":  m[0].cpu().numpy().astype(bool),
						"og":    og[0].cpu().numpy().astype(np.float32),   # float32, not Python floats
						"recon": rec[0].cpu().numpy().astype(np.float32),
					})
				examples[space] = rows
		out["examples"] = examples

	# ---------- save ----------
	if not test:
		test_name = test_params["test_name"]
		# per-spectrum losses: ~44 KB. Save for EVERY sweep member — the fixed split
		# makes these index-comparable across runs, which is what enables the
		# cross-model Spearman comparison later. Without them you'd have to retrain.
		np.savez_compressed(
			path.Path(test_name, f"{test_name}_{d_split}_losses.npz"),
			loss_scaled=loss_scaled, loss_unscaled=loss_unscaled,
		)
		if want_latent:
			save = {k: v for k, v in out.items()
			        if k in ("latent", "redshift", "snr", "loss_scaled", "loss_unscaled")
			        and v is not None}
			np.savez_compressed(path.Path(test_name, f"{test_name}_{d_split}_latent.npz"), **save)

	return out

def model_stats(outputs, test_params, best):

	test_name = test_params["test_name"]

	# all_losses_scaled = np.array([o["loss_scaled"] for o in outputs])
	all_losses_scaled = outputs["loss_scaled"]
	all_losses_unscaled = outputs["loss_unscaled"]

	# all_losses_unscaled = np.array([o["loss_unscaled"] for o in outputs])

	# all_rel_losses = np.array([o["rel_loss"] for o in outputs])

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
		# "rel": {
		# 	"mean":   float(np.mean(all_rel_losses)),
		# 	"median": float(np.median(all_rel_losses)),
		# 	"std":    float(np.std(all_rel_losses)),
		# 	"p95":    float(np.percentile(all_rel_losses, 95)),
		# 	"max":    float(np.max(all_rel_losses)),
		# },
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

# def global_stats(loader):

# 	all_fluxes = []
# 	for batch_flux, batch_mask in loader:
# 		mask = batch_flux != 0
# 		all_fluxes.append(batch_flux[mask])

# 	combined_fluxes = torch.cat(all_fluxes)

# 	return combined_fluxes.mean(), combined_fluxes.std()

def _to_physical_space(x_scaled, train_mean, train_std, standardize, flux_type, mask=None):

	# Step 1: invert z-score
	if standardize:
		x = (x_scaled * train_std) + train_mean
	else:
		x = x_scaled.clone()

	# Step 2: invert flux-space transform
	if flux_type == "log_scale_flux":
		x = torch.sign(x) * torch.expm1(torch.clamp(torch.abs(x), max=30.0)) # clampto not get any Nans/infs and mess everything up
	elif flux_type == "normalized_flux_cont":
		pass # not added yet
	elif flux_type == "normalized_flux_med":
		pass # not addded yet                          # raw flux, z-score inversion is sufficient
	else:
		logger.warning(f"Unknown flux_type '{flux_type}' — no flux-space inversion applied")

	if mask is not None:
		x = torch.where(mask, x, torch.zeros_like(x))
	return x

def log_final_stats(losses_per_epoch):

	valid_mse = losses_per_epoch["valid_mse"]
	valid_unscaled = losses_per_epoch["unscaled_valid_mses"]
	train_total = losses_per_epoch["train_total"]

	best_scaled = int(np.argmin(valid_mse))
	best_unscaled = int(np.argmin(valid_unscaled))

	wandb.log({
		"final/train_loss": train_total[-1],
		"final/valid_loss": losses_per_epoch["valid_total"][-1],
		#
		"final/best_valid_scaled":       min(losses_per_epoch["valid_total"]),
		"final/best_valid_unscaled_mse": min(valid_unscaled),   # sweep target
		"final/best_valid_log_mse":      min(valid_mse),
		#
		# ── the two selection epochs, like-for-like (both recon-only) ──
		"final/best_epoch_log":      best_scaled,
		"final/best_epoch_unscaled": best_unscaled,
		# if this is 0, the two metrics agree and the question is moot.
		# if it's large, they disagree and you have a choice to defend.
		"final/epoch_disagreement":  abs(best_scaled - best_unscaled),
	})

	wandb.log(_fit_check(losses_per_epoch))

	# wandb.log({
	# 	"final/train_loss": losses_per_epoch["train_total"][-1],
	# 	"final/valid_loss": losses_per_epoch["valid_total"][-1],
	# 	#
	# 	"final/best_valid_scaled": min(losses_per_epoch["valid_total"]),
	# 	"final/best_valid_unscaled_mse": min(losses_per_epoch["unscaled_valid_mses"]),  # target for wandb sweep
	# 	# "final/best_valid_unscaled_rel_mse": min(losses_per_epoch["unscaled_valid_rel_mses"]),
	# 	#
	# 	"final/best_epoch_scaled": int(np.argmin(losses_per_epoch["valid_total"])),
	# 	"final/train_at_best_scaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["valid_total"]))],
	# 	# "final/overfit_gap_scaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["valid_total"]))] - min(losses_per_epoch["valid_total"]),
	# 	#
	# 	"final/best_epoch_unscaled": int(np.argmin(losses_per_epoch["unscaled_valid_mses"])),
	# 	"final/train_at_best_unscaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["unscaled_valid_mses"]))],
	# 	#			
	# 	# "final/best_epoch_rel_unscaled": int(np.argmin(losses_per_epoch["unscaled_valid_rel_mses"])),
	# 	# "final/train_at_best_rel_unscaled": losses_per_epoch["train_total"][int(np.argmin(losses_per_epoch["unscaled_valid_rel_mses"]))],
	# })

	# wandb.log(_fit_check(losses_per_epoch))

def log_summary(train_outputs, valid_outputs, test_params, test = False):

	test_name = test_params["test_name"]

	# # all_losses_scaled = np.array([o["loss_scaled"] for o in outputs])

	# train_all_losses_unscaled = np.array([o["loss_unscaled"] for o in train_outputs])
	# train_all_losses_scaled = np.array([o["loss_scaled"] for o in train_outputs])
	# # train_all_rel_losses = np.array([o["rel_loss"] for o in train_outputs])

	# valid_all_losses_unscaled = np.array([o["loss_unscaled"] for o in valid_outputs])
	# valid_all_losses_scaled = np.array([o["loss_scaled"] for o in valid_outputs])
	# # valid_all_rel_losses = np.array([o["rel_loss"] for o in valid_outputs])

	# wandb.run.summary["mean_unscaled_mse"] = np.mean(valid_all_losses_unscaled)
	# # wandb.run.summary["mean_rel_mse"] = np.mean(valid_all_rel_losses) # unscaled i think...
	# # wandb.run.summary["mean_overfit"] = np.mean(train_all_rel_losses) - np.mean(valid_all_rel_losses)

	train_unscaled = train_outputs["loss_unscaled"]
	valid_unscaled = valid_outputs["loss_unscaled"]

	train_mse = float(np.mean(train_unscaled))
	valid_mse = float(np.mean(valid_unscaled))

	wandb.run.summary["mean_unscaled_mse"] = valid_mse
	# generalisation gap: valid - train. Positive is NORMAL; we care if it's LARGE.
	wandb.run.summary["gap"]          = valid_mse - train_mse
	wandb.run.summary["relative_gap"] = (valid_mse - train_mse) / max(abs(train_mse), 1e-12)

# def get_latent_space(loader, model, test_params, test=False):

# 	test_name = test_params["test_name"]

# 	device = next(model.parameters()).device

# 	ds = loader.dataset
# 	d_split = ds.split

# 	l = ds.l
# 	train_mean = ds.mean
# 	train_std = ds.std
# 	standardize = ds.standardize
# 	flux_type = ds.flux_type

# 	temp_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False)

# 	all_latent = []
# 	all_loss_scaled = []
# 	all_loss_unscaled = []
# 	# all_rel_loss = []

# 	model.eval()
# 	with torch.no_grad():
# 		for x, x_mask in temp_loader:

# 			x = x.to(device)
# 			x_mask = x_mask.to(device)

# 			z = model.encode(x)
# 			all_latent.append(z.cpu().numpy())

# 			x_hat, _, _ = model(x)

# 			all_loss_scaled.append(loss_calc_per_spec(x_hat, x, x_mask).cpu().numpy())

# 			x_unscaled = _to_physical_space(x, train_mean, train_std, standardize, flux_type, mask = x_mask)
# 			x_hat_unscaled = _to_physical_space(x_hat, train_mean, train_std, standardize, flux_type, mask = x_mask)


# 			# if flux_type == "log_scale_flux":
# 			# 	x_hat_unscaled = torch.sign(x_hat_unscaled) * torch.expm1(torch.abs(x_hat_unscaled))
# 			# 	x_unscaled = torch.sign(x_unscaled.to(device)) * torch.expm1(torch.abs(x_unscaled.to(device)))
# 			# x_hat_unscaled = x_hat_unscaled * x_mask
# 			# x_unscaled = x_unscaled.to(device) * x_mask

# 			all_loss_unscaled.append(loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask).cpu().numpy())
# 			# all_rel_loss.append(rel_loss_calc_per_spec(x_hat_unscaled, x_unscaled, x_mask).cpu().numpy())

# 	latent = np.concatenate(all_latent, axis=0)

# 	# --- latent health check ---------------------------------------------------
# 	# A unit that is identically zero across the whole split contributes nothing:
# 	# your EFFECTIVE latent size is smaller than latent_size says. This is a real
# 	# risk with a ReLU bottleneck (dead units never recover: output 0 -> grad 0).
# 	# It should be 0 with the linear bottleneck — log it to PROVE that, not to fix it.
# 	unit_std = latent.std(axis=0)
# 	n_dead   = int((unit_std < 1e-8).sum())
# 	n_latent = latent.shape[1]

# 	logger.info(f"[{d_split}] latent: {n_dead}/{n_latent} dead units, "
# 				f"{n_latent - n_dead} effective")
# 	if n_dead:
# 		logger.warning(f"[{d_split}] {n_dead} dead latent units — "
# 					   f"effective latent size is {n_latent - n_dead}, not {n_latent}")

# 	wandb.run.summary[f"{d_split}/n_dead_latent"]  = n_dead
# 	wandb.run.summary[f"{d_split}/n_eff_latent"]   = n_latent - n_dead
# 	# ---------------------------------------------------------------------------

# 	loss_scaled = np.concatenate(all_loss_scaled)
# 	loss_unscaled = np.concatenate(all_loss_unscaled)
# 	# rel_loss = np.concatenate(all_rel_loss)

# 	redshift = temp_loader.dataset._get_redshift()
# 	snr = temp_loader.dataset._get_snr()

# 	latent_data = {
# 		"latent":        latent,
# 		"loss_scaled":   loss_scaled,
# 		"loss_unscaled": loss_unscaled,
# 		# "rel_loss":      rel_loss,
# 		"redshift":      redshift,
# 		"snr":           snr,
# 	}

# 	if not test:
# 		save_arrays = {k: v for k, v in latent_data.items() if v is not None}
# 		pth = path.Path(test_name, f"{test_name}_{d_split}_latent.npz")
# 		np.savez(pth, **save_arrays)

# 	return latent_data

def _fit_check(losses):
	
	"""Sanity flag on the selected model. NOT an object of study.
	A positive gap is normal — every model has one. We only care if it's LARGE."""
	train = np.asarray(losses["train_total"]) # scaled loss
	valid = np.asarray(losses["valid_total"]) # scaled loss
	best  = int(np.argmin(losses["valid_mse"]))   

	gap = float(valid[best] - train[best])                  # valid - train. Positive = normal.
	rel = gap / max(abs(float(train[best])), 1e-12)         # scale-free -> comparable across runs

	return {"fit/gap": gap, "fit/relative_gap": rel, "fit/best_epoch": best}

