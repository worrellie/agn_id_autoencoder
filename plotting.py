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

from sklearn.manifold import TSNE
try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

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
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

	# plot total loss (mse + kl)
	ax1.plot(epochs, train_loss, label="Train Total", alpha=0.8)
	ax1.plot(epochs, valid_loss, label="Valid Total", alpha=0.8, linestyle="--")
	ax1.set_title(f"Total Loss (Beta={model_losses['beta']})")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss Value")
	ax1.legend()

	# plot mse and kl divergence separately
	ax2.plot(epochs, train_mse, label="Train MSE", color="tab:blue")
	ax2.plot(epochs, valid_mse, label="Valid MSE", color="tab:blue", linestyle="--")

	# if kl diference is use (vae)
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

	# plot rel mse in unscaled space to see if model converges in metric of interest
	rel_mse = model_losses["unscaled_valid_rel_mses"]
	print(rel_mse)
	print(train_mse)
	print(valid_mse)
	exit()
	ax3.plot(epochs, rel_mse, label="Valid Rel MSE (unscaled)", color="tab:green")
	best_epoch = int(np.argmin(rel_mse))
	ax3.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.6,
				label=f"Best epoch ({best_epoch})")
	ax3.set_title("Relative MSE — Physical Space (Sweep Target)")
	ax3.set_xlabel("Epochs")
	ax3.set_ylabel("Rel MSE")
	ax3.legend()

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

def plot_log_vs_rel_mse(model_losses, test_params, test=False):

	test_name  = test_params["test_name"]
	# epochs     = range(1, len(model_losses["valid_mse"]) + 1)
	epochs     = range(0, len(model_losses["valid_mse"]))
	log_mse    = model_losses["valid_mse"]
	rel_mse    = model_losses["unscaled_valid_rel_mses"]

	fig, ax1 = plt.subplots(figsize=(10, 4))
	c_log, c_rel = "tab:orange", "tab:green"

	ax1.plot(epochs, log_mse, color=c_log, label="Scaled MSE")
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Scaled MSE", color=c_log)
	ax1.tick_params(axis="y", labelcolor=c_log)

	ax2 = ax1.twinx()
	ax2.plot(epochs, rel_mse, color=c_rel, label="Rel MSE (unscaled)")
	ax2.set_ylabel("Rel MSE (unscaled)", color=c_rel)
	ax2.tick_params(axis="y", labelcolor=c_rel)

	best_log = int(np.argmin(log_mse))
	best_rel = int(np.argmin(rel_mse))
	ax1.axvline(best_log, color=c_log, linestyle="--", alpha=0.5, label=f"Best scaled (ep {best_log})")
	ax2.axvline(best_rel, color=c_rel, linestyle="--", alpha=0.5, label=f"Best rel (ep {best_rel})")

	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
	plt.title("Scaled MSE vs Physical Rel MSE per Epoch")
	plt.tight_layout()

	if not test:
		pth_fig = path.Path(test_name, f"{test_name}_log_vs_rel_mse.png")
		fig.savefig(pth_fig)
		wandb.log({"metrics/log_vs_rel_mse": wandb.Image(fig)})
		plt.close(fig)
	else:
		plt.show()

	return fig


def plot_latent_space(latent_data, color_by, color_label=None,
                      method="tsne", test_params=None, test=False):
	"""
	Project the latent space to 2D and colour points by a chosen parameter.

	Parameters
	----------
	latent_data : dict
	    Output of funcs.get_latent_space(). Must contain key "latent" (N, D).
	    May also contain "redshift", "snr", "loss_scaled", "loss_unscaled", "rel_loss".
	color_by : str or np.ndarray
	    Key in latent_data to use as the colour axis, or a raw array of length N.
	color_label : str, optional
	    Colorbar label. Defaults to color_by if a string was given.
	method : {"tsne", "umap", "both"}
	    Dimensionality reduction method(s) to use.
	test_params : dict, optional
	    Must contain "test_name" when test=False so plots can be saved.
	test : bool
	    If True, display interactively instead of saving.
	"""

	latent = latent_data["latent"]

	if isinstance(color_by, str):
		if color_label is None:
			color_label = color_by
		c = latent_data.get(color_by)
		if c is None:
			raise ValueError(f"color_by='{color_by}' not found in latent_data. "
			                 f"Available keys: {list(latent_data.keys())}")
		c = np.array(c, dtype=float)
	else:
		c = np.array(color_by, dtype=float)
		if color_label is None:
			color_label = "value"

	if method == "both" and not _UMAP_AVAILABLE:
		raise ImportError("umap-learn is not installed. Install it with: pip install umap-learn")
	if method == "umap" and not _UMAP_AVAILABLE:
		raise ImportError("umap-learn is not installed. Install it with: pip install umap-learn")

	# print(len(latent))
	# print(min(30, len(latent) - 1))
	# print(latent)

	def _reduce_tsne(z):
		return TSNE(n_components=2, perplexity=min(30, len(z) - 1)).fit_transform(z)

	def _reduce_umap(z):
		reducer = umap.UMAP(n_components=2,)
		return reducer.fit_transform(z)

	plt.style.use("fivethirtyeight")

	if method == "both":
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
		axes_methods = [(ax1, "t-SNE", _reduce_tsne), (ax2, "UMAP", _reduce_umap)]
	else:
		fig, ax = plt.subplots(figsize=(9, 7))
		reducer = _reduce_tsne if method == "tsne" else _reduce_umap
		label = "t-SNE" if method == "tsne" else "UMAP"
		axes_methods = [(ax, label, reducer)]

	for ax, label, reducer in axes_methods:
		embedding = reducer(latent)
		sc = ax.scatter(embedding[:, 0], embedding[:, 1],
		                c=c, cmap="viridis", s=4, alpha=0.6, rasterized=True)
		plt.colorbar(sc, ax=ax, label=color_label)
		ax.set_title(f"Latent space — {label}")
		ax.set_xlabel(f"{label} 1")
		ax.set_ylabel(f"{label} 2")

	plt.tight_layout()

	figs = {}
	if not test and test_params is not None:
		test_name = test_params["test_name"]
		stem = f"{test_name}_latent_{method}_{color_label.replace(' ', '_')}"
		pth_fig = path.Path(test_name, f"{stem}.png")
		pth_obj = path.Path(test_name, f"{stem}.pkl")
		fig.savefig(pth_fig, dpi=150)
		with open(pth_obj, "wb") as o:
			pkl.dump(fig, o)
	else:
		plt.show()

	return fig