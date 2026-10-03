"""Plot the contents of a finished run folder.

Usage:
    uv run python plot_run.py RUN_DIR                  # make every plot
    uv run python plot_run.py RUN_DIR loss_vs_snr      # make selected plots
    uv run python plot_run.py RUN_DIR --zmax 1.8       # cut both samples to z <= 1.8 first
    uv run python plot_run.py --list                   # list available plots

Figures are saved to RUN_DIR/plots/.

In a notebook:
    from plot_run import Run, PLOTS
    run = Run("RUN_...")
    fig = PLOTS["loss_vs_snr"](run)

ADDING A NEW PLOT: write a function that takes a Run and returns a figure, and put
@plot("its_name") above it. It then appears in --list and runs with everything else.
Data available on a Run:
    run.params        dict from _params.json (flux_type, latent_size, norm_stats, ...)
    run.epochs        dict from _losses.json (train_mse, valid_mse, unscaled_valid_mses, ...)
    run.normal        normal-galaxy validation: loss_scaled, loss_unscaled, latent, snr, redshift
    run.agn           AGN validation: same keys
    run.split(name)   any other saved split, e.g. run.split("train") in full-analysis runs
    run.model()       (model, norm_stats, params) from the best checkpoint; needs torch + repo code
"""
import argparse
import json
import logging
import pathlib
from functools import cached_property

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
NORMAL_C, AGN_C = "royalblue", "crimson"

# ─────────────────────────── loading ───────────────────────────


class Run:
    """Lazy access to the files in one run folder. Nothing is read until used."""

    def __init__(self, run_dir, zmax=None):
        self.dir = pathlib.Path(run_dir)
        self.name = self.dir.name
        self.zmax = zmax
        self._splits = {}

    def file(self, suffix):
        return self.dir / f"{self.name}_{suffix}"

    @cached_property
    def params(self):
        return json.loads(self.file("params.json").read_text())

    @cached_property
    def epochs(self):
        return json.loads(self.file("losses.json").read_text())

    def split(self, name):
        """Per-spectrum arrays for a saved split ('validation', 'agn_validation', 'train').
        Prefers the _latent.npz (has latent, snr, redshift); falls back to _losses.npz."""
        if name not in self._splits:
            for suffix in (f"{name}_latent.npz", f"{name}_losses.npz"):
                f = self.file(suffix)
                if f.exists():
                    with np.load(f) as npz:
                        d = {k: npz[k] for k in npz.files}
                    break
            else:
                raise FileNotFoundError(f"No saved arrays for split '{name}' in {self.dir}")
            if self.zmax is not None and "redshift" in d:
                keep = d["redshift"] <= self.zmax
                d = {k: v[keep] for k, v in d.items()}
            self._splits[name] = d
        return self._splits[name]

    @property
    def normal(self):
        return self.split("validation")

    @property
    def agn(self):
        return self.split("agn_validation")

    def model(self, device="cpu", which="best"):
        import funcs  # repo code; only imported when a plot needs the model
        return funcs.load_model(self.file(f"{which}_model.pt"), device)

    @property
    def tag(self):
        """Short suffix for filenames/titles describing any data cuts."""
        return f"_z{self.zmax:g}" if self.zmax is not None else ""


# ─────────────────────────── registry ───────────────────────────

PLOTS = {}


def plot(name):
    def register(fn):
        PLOTS[name] = fn
        return fn
    return register


def _title(fig, run, what):
    cut = f"  [z ≤ {run.zmax:g}]" if run.zmax is not None else ""
    fig.suptitle(f"{what}{cut}\n{run.name}", fontsize=9)


def _log_bins(*arrays, n=60):
    x = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays])
    return np.logspace(np.log10(x.min()), np.log10(x.max()), n + 1)


def _binned_median(x, y, bins, min_count=20):
    idx = np.digitize(x, bins) - 1
    centres, medians = [], []
    for i in range(len(bins) - 1):
        sel = idx == i
        if sel.sum() >= min_count:
            centres.append(np.sqrt(bins[i] * bins[i + 1]))
            medians.append(np.median(y[sel]))
    return np.array(centres), np.array(medians)


SPACES = {"scaled": "Reconstruction MSE (log-scaled, standardised)",
          "unscaled": "Reconstruction MSE (physical / normalised flux)"}

# ─────────────────────────── plots ───────────────────────────


@plot("training_curves")
def training_curves(run):
    e = run.epochs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = np.arange(1, len(e["train_mse"]) + 1)
    axes[0].plot(ep, e["train_mse"], label="Train")
    axes[0].plot(ep, e["valid_mse"], label="Validation")
    axes[0].set_ylabel("MSE (scaled)")
    axes[1].plot(ep, e["unscaled_valid_mses"], color="C1", label="Validation")
    axes[1].set_ylabel("MSE (unscaled)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
    _title(fig, run, "Loss during training")
    fig.tight_layout()
    return fig


@plot("agn_vs_normal_hist")
def agn_vs_normal_hist(run):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (space, xlabel) in zip(axes, SPACES.items()):
        n, a = run.normal[f"loss_{space}"], run.agn[f"loss_{space}"]
        bins = _log_bins(n, a)
        ax.hist(n, bins=bins, alpha=0.6, color=NORMAL_C, label=f"Normal galaxies (N={len(n)})")
        ax.hist(a, bins=bins, alpha=0.6, color=AGN_C, label=f"AGN (N={len(a)})")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("N")
        ax.legend()
        ax.grid(alpha=0.3)
    _title(fig, run, "Reconstruction loss: normal vs AGN (validation)")
    fig.tight_layout()
    return fig


def _loss_vs(run, key, xlabel, what):
    """Scatter of loss against a per-spectrum quantity, with binned medians.
    The medians are the like-for-like comparison: AGN vs normal at the same x."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (space, ylabel) in zip(axes, SPACES.items()):
        for d, colour, label in ((run.normal, NORMAL_C, "Normal"), (run.agn, AGN_C, "AGN")):
            x, y = d[key], d[f"loss_{space}"]
            ax.scatter(x, y, s=2, alpha=0.15, color=colour, rasterized=True)
            bins = _log_bins(x, n=15) if key == "snr" else np.linspace(x.min(), x.max(), 16)
            c, m = _binned_median(x, y, bins)
            ax.plot(c, m, "o-", color=colour, lw=2, label=f"{label} (binned median)")
        if key == "snr":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    _title(fig, run, what)
    fig.tight_layout()
    return fig


@plot("loss_vs_snr")
def loss_vs_snr(run):
    return _loss_vs(run, "snr", "SNR per pixel (SNR_MEAN or SNR_MED, matching flux_type)", "Reconstruction loss vs SNR")


@plot("loss_vs_redshift")
def loss_vs_redshift(run):
    return _loss_vs(run, "redshift", "Redshift", "Reconstruction loss vs redshift")


@plot("latent_pca")
def latent_pca(run):
    """One shared 2D projection (PCA fitted on normal galaxies) shown in four colourings."""
    zn, za = run.normal["latent"], run.agn["latent"]
    mu = zn.mean(axis=0)
    _, _, vt = np.linalg.svd(zn - mu, full_matrices=False)
    pn, pa = (zn - mu) @ vt[:2].T, (za - mu) @ vt[:2].T
    both = np.vstack([pn, pa])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    ax.scatter(pn[:, 0], pn[:, 1], s=2, alpha=0.3, color=NORMAL_C, label="Normal", rasterized=True)
    ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color=AGN_C, label="AGN", rasterized=True)
    ax.legend(markerscale=5)
    ax.set_title("Sample")

    colourings = [("snr", "SNR (log)", True), ("redshift", "Redshift", False),
                  ("loss_scaled", "Scaled loss (log)", True)]
    for ax, (key, label, log) in zip(axes.flat[1:], colourings):
        c = np.concatenate([run.normal[key], run.agn[key]])
        c = np.log10(np.clip(c, 1e-12, None)) if log else c
        sc = ax.scatter(both[:, 0], both[:, 1], c=c, s=2, alpha=0.4, cmap="viridis", rasterized=True)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(label)
    for ax in axes.flat:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    _title(fig, run, f"Latent space, PCA (latent size {zn.shape[1]})")
    fig.tight_layout()
    return fig


@plot("reconstructions")
def reconstructions(run, device="cpu", n=4):
    """Original vs reconstruction (scaled space) for normal and AGN spectra at
    evenly spaced loss percentiles. Loads the best checkpoint and the data files."""
    import torch
    from datahandling import H5SpecDataset

    if run.zmax is not None:
        raise RuntimeError("reconstructions uses file row indices, so run it without --zmax")
    model, norm_stats, params = run.model(device)
    sets = [("Normal", params["data_file"], run.split("validation")),
            ("AGN", params["agn_data"], run.split("agn_validation"))]

    pct = np.linspace(5, 95, n)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.6 * n), sharex=True)
    for col, (label, data_file, d) in enumerate(sets):
        ds = H5SpecDataset(data_file, split="validation", flux_type=params["flux_type"],
                           standardize=params["standardize"], norm_stats=norm_stats,
                           preload="none", device=device)
        loss = d["loss_scaled"]
        for row, p in enumerate(pct):
            i = int(np.argmin(np.abs(loss - np.percentile(loss, p))))
            x, m = ds[i]
            with torch.no_grad():
                x_hat, _, _ = model(x.unsqueeze(0).to(device))
            x, x_hat, m = x.numpy(), x_hat[0].cpu().numpy(), m.numpy().astype(bool)
            x = np.where(m, x, np.nan)
            ax = axes[row, col]
            ax.plot(ds.l, x, lw=0.5, color="grey", label="Original")
            ax.plot(ds.l, np.where(m, x_hat, np.nan), lw=0.8,
                    color=NORMAL_C if label == "Normal" else AGN_C, label="Reconstruction")
            ax.set_title(f"{label}: {p:.0f}th pct loss = {loss[i]:.3g}, SNR = {d['snr'][i]:.2g}",
                         fontsize=8)
            ax.grid(alpha=0.3)
        axes[0, col].legend(fontsize=7)
        axes[-1, col].set_xlabel("Wavelength (Å)")
    _title(fig, run, "Reconstructions (scaled space)")
    fig.tight_layout()
    return fig


# ─────────────────────────── CLI ───────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("plots", nargs="*", help="plot names (default: all)")
    parser.add_argument("--zmax", type=float, default=None, help="keep only z <= zmax")
    parser.add_argument("--list", action="store_true", help="list available plots")
    parser.add_argument("--fmt", default="png", help="png or pdf")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list or not args.run_dir:
        print("\n".join(PLOTS))
        return

    run = Run(args.run_dir, zmax=args.zmax)
    out_dir = run.dir / "plots"
    out_dir.mkdir(exist_ok=True)

    for name in args.plots or PLOTS:
        if name not in PLOTS:
            logger.error(f"unknown plot '{name}' — see --list")
            continue
        try:
            fig = PLOTS[name](run)
        except Exception as e:  # one missing file shouldn't stop the other plots
            logger.warning(f"skipped {name}: {type(e).__name__}: {e}")
            continue
        out = out_dir / f"{name}{run.tag}.{args.fmt}"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"saved {out}")


if __name__ == "__main__":
    main()