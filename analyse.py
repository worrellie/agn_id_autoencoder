import argparse, json, logging
import pathlib as path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import autoencoder as ae
import funcs
import plotting
from datahandling import make_datasets, make_dataloader

logger = logging.getLogger(__name__)


def load_run(run_dir, device, preload="auto", batch_size=64):
	"""params.json + best_model.pt -> (model, train_loader, valid_loader, params)"""
	run_dir = path.Path(run_dir)
	name = run_dir.name
	
	if not (run_dir / f"{name}_params.json").exists():
		raise FileNotFoundError(
			f"{run_dir}/{name}_params.json not found. "
			f"Run analyse.py from the PARENT directory of the run folder."
		)


	with open(run_dir / f"{name}_params.json") as f:
		params = json.load(f)

	datasets, mode = make_datasets(
		params["data_file"], splits=["train", "validation"],
		flux_type=params["flux_type"], standardize=params["standardize"],
		device=device, preload_request=preload,
	)
	logger.info(f"[{name}] preload -> {mode}")

	train_loader = make_dataloader(datasets["train"],      batch_size=batch_size, shuffle=False)
	valid_loader = make_dataloader(datasets["validation"], batch_size=batch_size, shuffle=False)

	ckpt = torch.load(run_dir / f"{name}_best_model.pt", map_location=device)
	model = ae.build_model(params, state_dict=ckpt["model"], device=device)

	if "epoch" in ckpt:   # self-describing checkpoint (if you took option (b))
		logger.info(f"[{name}] best epoch {ckpt['epoch']}, {ckpt.get('metric_name')}={ckpt.get('metric'):.6g}")

	return model, train_loader, valid_loader, params


def analyse_one(run_dir, device, method="both"):
	"""Everything the `full` tier does, but on a saved run. No retraining."""
	model, train_loader, valid_loader, params = load_run(run_dir, device)
	test_params = params
	l = valid_loader.dataset.l

	valid_ev = funcs.evaluate(valid_loader, model, test_params, want_latent=True, want_examples=True)
	train_ev = funcs.evaluate(train_loader, model, test_params, want_latent=True, want_examples=True)

	# ---- the scaling comparison (your point 2) ------------------------------
	ls, lu = valid_ev["loss_scaled"], valid_ev["loss_unscaled"]
	rho, _ = spearmanr(ls, lu)
	pearson, _ = pearsonr(ls, lu)
	top_log  = set(np.argsort(ls)[-100:])
	top_phys = set(np.argsort(lu)[-100:])
	overlap = len(top_log & top_phys)

	print(f"\n=== {path.Path(run_dir).name} ===")
	print(f"  Spearman(log-space, physical-space) per-spectrum loss : {rho:.3f}")
	print(f"  Pearson(log-space, physical-space) per-spectrum loss : {pearson:.3f}")
	print(f"  top-100 anomaly overlap between the two metrics       : {overlap}/100")
	print(f"  effective latent size : {valid_ev['n_eff']} / {valid_ev['latent'].shape[1]}")
	print(f"  valid loss p95/median : {np.percentile(lu, 95) / np.median(lu):.2f}   "
		  f"(separation ratio — a FLAT distribution means no discrimination)")

	# ---- plots -------------------------------------------------------------
	plotting.plot_dists(train_ev, valid_ev, test_params)
	plotting.plot_examples(valid_ev, l, test_params)
	plotting.plot_examples(train_ev, l, test_params)

	color_params = [("loss_scaled", "Scaled loss"), ("loss_unscaled", "Unscaled MSE"),
					("redshift", "Redshift"), ("snr", "SNR")]
	methods = ("tsne", "umap") if method == "both" else (method,)
	for m in methods:
		plotting.plot_latent_panels(valid_ev, color_params, m, test_params)

	# ---- anomaly candidates: the actual point of the whole pipeline ---------
	top = np.argsort(lu)[::-1][:50]
	np.savez(path.Path(run_dir) / f"{path.Path(run_dir).name}_anomaly_candidates.npz",
			 indices=top, loss=lu[top],
			 redshift=valid_ev["redshift"][top] if valid_ev["redshift"] is not None else [],
			 snr=valid_ev["snr"][top] if valid_ev["snr"] is not None else [])

	plt.close("all")
	return valid_ev


def compare(run_dirs, device):
	"""
	Cross-model. Relies on the FIXED train/valid split: spectrum i is the same
	galaxy in every run, so per-spectrum losses are index-comparable.
	"""
	evs = {}
	for r in run_dirs:
		name = path.Path(r).name
		npz = path.Path(r) / f"{name}_validation_losses.npz"
		if npz.exists():                        # cheap path — saved by every sweep member
			evs[name] = np.load(npz)["loss_unscaled"]
		else:                                   # fall back to recomputing
			model, _, valid_loader, params = load_run(r, device)
			evs[name] = funcs.evaluate(valid_loader, model, params)["loss_unscaled"]

	names = list(evs)

	# --- agreement matrix ---------------------------------------------------
	print("\n=== Do the models agree on WHICH spectra are anomalous? ===")
	print("(high rho => the anomaly ranking is a property of the DATA, not the architecture)")
	for i, a in enumerate(names):
		for b in names[i + 1:]:
			rho, _ = spearmanr(evs[a], evs[b])
			ta = set(np.argsort(evs[a])[-100:])
			tb = set(np.argsort(evs[b])[-100:])
			print(f"  {a:28s} vs {b:28s}  rho={rho:.3f}  top100 overlap={len(ta & tb)}/100")

	# --- overlaid loss distributions: THE tail is the product ---------------
	fig, ax = plt.subplots(figsize=(9, 5))
	for n in names:
		ax.hist(np.log10(evs[n] + 1e-12), bins=80, histtype="step", lw=1.6, label=n)
	ax.set_xlabel("log10(per-spectrum validation MSE)")
	ax.set_ylabel("N")
	ax.set_title("Loss distributions — look for a long RIGHT TAIL, not a low mean")
	ax.legend(fontsize=7)
	plt.tight_layout()
	fig.savefig("compare_loss_distributions.png", dpi=150)
	print("\nwrote compare_loss_distributions.png")


if __name__ == "__main__":

	logging.basicConfig(level=logging.INFO)
	p = argparse.ArgumentParser()
	p.add_argument("runs", nargs="+", help="run directories")
	p.add_argument("--compare", action="store_true", help="cross-model comparison only")
	p.add_argument("--method", default="both", choices=["tsne", "umap", "both"])
	args = p.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if args.compare:
		compare(args.runs, device)
	else:
		for r in args.runs:
			analyse_one(r, device, method=args.method)