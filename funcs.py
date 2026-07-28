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

			x_unscaled     = _to_norm_space(x,     train_mean, train_std, standardize, flux_type, mask=x_mask)
			x_hat_unscaled = _to_norm_space(x_hat, train_mean, train_std, standardize, flux_type, mask=x_mask)
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
						og  = _to_norm_space(x,     train_mean, train_std, standardize, flux_type, mask=m)
						rec = _to_norm_space(x_hat, train_mean, train_std, standardize, flux_type, mask=m)

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

def _to_norm_space(x_scaled, train_mean, train_std, standardize, flux_type, mask=None):

	# inverse transform spectra to NORMALISED space
	# ie de-standardize and de-log

	# Step 1: invert z-score
	if standardize:
		x = (x_scaled * train_std) + train_mean
	else:
		x = x_scaled.clone()

	# Step 2: invert flux-space transform
	if flux_type == "log_scale_flux":
		x = torch.sign(x) * torch.expm1(torch.clamp(torch.abs(x), max=30.0)) # clampto not get any Nans/infs and mess everything up
	if  flux_type == "log_scale_flux_med":
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

def log_summary(train_outputs, valid_outputs, test_params, test = False):

	test_name = test_params["test_name"]

	train_unscaled = train_outputs["loss_unscaled"]
	valid_unscaled = valid_outputs["loss_unscaled"]

	train_mse = float(np.mean(train_unscaled))
	valid_mse = float(np.mean(valid_unscaled))

	wandb.run.summary["mean_unscaled_mse"] = valid_mse
	# generalisation gap: valid - train. Positive is NORMAL; we care if it's LARGE.
	wandb.run.summary["gap"]          = valid_mse - train_mse
	wandb.run.summary["relative_gap"] = (valid_mse - train_mse) / max(abs(train_mse), 1e-12)

def _fit_check(losses):
	
	"""Sanity flag on the selected model. NOT an object of study.
	A positive gap is normal — every model has one. We only care if it's LARGE."""
	train = np.asarray(losses["train_total"]) # scaled loss
	valid = np.asarray(losses["valid_total"]) # scaled loss
	best  = int(np.argmin(losses["valid_mse"]))   

	gap = float(valid[best] - train[best])                  # valid - train. Positive = normal.
	rel = gap / max(abs(float(train[best])), 1e-12)         # scale-free -> comparable across runs

	return {"fit/gap": gap, "fit/relative_gap": rel, "fit/best_epoch": best}

