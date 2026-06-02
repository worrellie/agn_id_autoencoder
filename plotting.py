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

def plot_loss_epoch_avg(model_losses, test_params, test=False):

	test_name = test_params["test_name"]

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
		return fig
	else:
		plt.show()



def plot_dists(train_outputs, valid_outputs, test_params, test=False):

	test_name = test_params["test_name"]

	train_l_scaled = np.array([o["loss_scaled"] for o in train_outputs])
	valid_l_scaled = np.array([o["loss_scaled"] for o in valid_outputs])

	train_l_unscaled = np.array([o["loss_unscaled"] for o in train_outputs])
	valid_l_unscaled = np.array([o["loss_unscaled"] for o in valid_outputs])

	train_l_rel = np.array([o["rel_loss"] for o in train_outputs])
	valid_l_rel = np.array([o["rel_loss"] for o in valid_outputs])

	fig, axes = plt.subplots(1, 3, figsize=(24, 4))

	_plot_dist(train_l_scaled, valid_l_scaled, axes[0], title="Loss Distribution (Scaled Space)")
	_plot_dist(train_l_unscaled, valid_l_unscaled,axes[1], title="Loss Distribution (Unscaled Space)")
	_plot_dist(train_l_rel, valid_l_rel,axes[2], title="Loss Distribution (Relative)")

	plt.tight_layout()

	if not test:
		pth_fig = path.Path(test_name, f"{test_name}_dists.png")
		pth_obj = path.Path(test_name, f"{test_name}_dists.pkl")
		plt.savefig(pth_fig)
		with open(pth_obj, "wb") as o:
			pkl.dump(fig, o)

		return fig
	else:
		plt.show()
	
	

def _plot_dist(train_losses, valid_losses, ax, title="Loss Distribution"):

	ax.hist(train_losses, bins=50, color='steelblue', edgecolor='white', label="Train")
	ax.hist(valid_losses, bins=50, color='tomato', edgecolor='white', label="Valid")

	ax.set_xlabel('Reconstruction Loss (MSE)')
	ax.set_ylabel('N')
	ax.set_title(title)

	ax.legend()

	# plt.tight_layout()

	return ax

def plot_examples(outputs, l, test_params, test=False):

	test_name = test_params["test_name"]

	all_losses_scaled = np.array([o["loss_scaled"] for o in outputs])

	all_losses_unscaled = np.array([o["loss_unscaled"] for o in outputs])

	all_rel_losses = np.array([o["rel_loss"] for o in outputs])

	targets_scaled = {
	"min":  np.min(all_losses_scaled),
	"25th": np.percentile(all_losses_scaled, 25),
	"mean": np.mean(all_losses_scaled),
	"75th": np.percentile(all_losses_scaled, 75),
	"max":  np.max(all_losses_scaled),
	}

	targets_unscaled = {
	"min":  np.min(all_losses_unscaled),
	"25th": np.percentile(all_losses_unscaled, 25),
	"mean": np.mean(all_losses_unscaled),
	"75th": np.percentile(all_losses_unscaled, 75),
	"max":  np.max(all_losses_unscaled),
	}

	targets_rel = {
	"min":  np.min(all_rel_losses),
	"25th": np.percentile(all_rel_losses, 25),
	"mean": np.mean(all_rel_losses),
	"75th": np.percentile(all_rel_losses, 75),
	"max":  np.max(all_rel_losses),
	}

	# examples
	examples_scaled = [dict(outputs[int(np.argmin(np.abs(all_losses_scaled - t)))], label=lbl) for lbl, t in targets_scaled.items()]
	examples_unscaled = [dict(outputs[int(np.argmin(np.abs(all_losses_unscaled - t)))], label=lbl) for lbl, t in targets_unscaled.items()]
	examples_rel = [dict(outputs[int(np.argmin(np.abs(all_rel_losses - t)))], label=lbl) for lbl, t in targets_rel.items()]

	plot_configs = [
	("scaled", examples_scaled, all_losses_scaled,   "loss_scaled"),
	("unscaled", examples_unscaled, all_losses_unscaled, "loss_unscaled"),
	("rel", examples_rel, all_rel_losses, "rel_loss"),
	]

	figs = {}
	for space, examples, _, loss_key in plot_configs:
		fig, axes = plt.subplots(5, 2, figsize=(16, 20))
		fig.suptitle(
			f"{space.upper()} — latent: {test_params['latent_size']}, "
			f"{test_params['activation_function']}, epochs: {test_params['max_epochs']}"
		)

		og_key    = "original_scaled"   if space == "scaled" else \
					"original_unscaled" if space == "unscaled" else \
					"original_unscaled"
		recon_key = "recon_scaled"      if space == "scaled" else \
					"recon_unscaled"    if space == "unscaled" else \
					"recon_unscaled"

		for ax_row, ex in zip(axes, examples):
			ax_fit, ax_res = ax_row
			og    = np.array(ex[og_key],    dtype=float)
			recon = np.array(ex[recon_key], dtype=float)
			mask  = np.array(ex["mask"],    dtype=bool)

			og[~mask]    = np.nan
			recon[~mask] = np.nan
			resid = og - recon

			ax_fit.step(l, og, color="black", lw=1.5, alpha=0.7, where="mid", label="Original")
			ax_fit.step(l, recon, color="red",   lw=1.0,            where="mid", label="Reconstructed")
			ax_fit.set_title(f"{ex['label']},  {loss_key}: {ex[loss_key]:.5f}")
			ax_fit.legend(fontsize=8)
			ax_fit.set_ylabel("Flux")

			ax_res.scatter(l, resid, color="gray", s=2)
			ax_res.axhline(0, color="black", lw=0.8, ls=":")
			ax_res.set_ylabel("Residual")
			ax_res.set_xlabel("Wavelength")

		plt.tight_layout()
		figs[space] = fig

		if not test:
			pth_fig = path.Path(test_name, f"{test_name}_recon_{space}.png")
			pth_obj = path.Path(test_name, f"{test_name}_recon_{space}.pkl")
			fig.savefig(pth_fig)
			with open(pth_obj, "wb") as o:
				pkl.dump(fig, o)
		else:
			plt.show()

	scaled_fig = figs['scaled']
	unscaled_fig = figs['unscaled']
	rel_fig = figs['rel']

	return scaled_fig, unscaled_fig, rel_fig
