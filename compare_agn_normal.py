"""Model-free comparison of the normal-galaxy and AGN validation data.

After 2 epochs the autoencoder is close to predicting the average spectrum, so its
loss is roughly each spectrum's spread around the training mean. This script
computes that directly from the files, with no model involved. If AGN come out
~10x lower here too, the difference is in the data/processing, not the model.

Usage:
    uv run python compare_agn_normal.py all_spectra_float32_v3.h5 all_agn_float32.h5 [log_scale_flux]
"""
import sys
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt

from datahandling import load_norm_stats

warnings.filterwarnings("ignore", message="All-NaN slice")
normal_file, agn_file = sys.argv[1:3]
flux_type = sys.argv[3] if len(sys.argv) > 3 else "log_scale_flux"
snr_key = "SNR_MED" if flux_type.endswith("_med") else "SNR_MEAN"
ns = load_norm_stats(normal_file, flux_type)   # training stats, as the model uses


def summarise(path, label):
    with h5py.File(path, "r") as hf:
        g = hf["validation"]
        x = g[flux_type][:].astype(np.float64)
        z = (x - ns["mean"]) / ns["std"]
        out = {
            "label": label,
            "x": x,
            "N": len(x),
            "mean z^2 per spectrum (median)": np.nanmedian(np.nanmean(z**2, axis=1)),
            "flux std within spectrum (median)": np.nanmedian(np.nanstd(x, axis=1)),
            "flux mean (all pixels)": np.nanmean(x),
            "flux 1st / 99th percentile": "{:.3f} / {:.3f}".format(*np.nanpercentile(x, [1, 99])),
            "NaN fraction": np.isnan(x).mean(),
            f"{snr_key} (median)": np.median(g[snr_key][:]) if snr_key in g else "n/a",
            "NORM_CMN (median)": np.median(g["NORM_CMN"][:]) if "NORM_CMN" in g else "n/a",
            "NOISE (median)": np.median(g["NOISE"][:]) if "NOISE" in g else "n/a",
            "redshift range": "{:.3f} - {:.3f}".format(g["redshift"][:].min(), g["redshift"][:].max()),
        }
        wl = hf.attrs["wavelengths"][:]
    return out, wl


normal, wl = summarise(normal_file, "Normal")
agn, _ = summarise(agn_file, "AGN")

print(f"flux_type={flux_type}, training mean={ns['mean']:.4f}, std={ns['std']:.4f}\n")
print(f"{'':38s}{'Normal':>22s}{'AGN':>22s}")
for key in normal:
    if key in ("label", "x"):
        continue
    a, b = normal[key], agn[key]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    print(f"{key:38s}{fmt(a):>22s}{fmt(b):>22s}")

# a few random spectra from each, plus each set's median spectrum
rng = np.random.default_rng(0)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for ax, d, colour in zip(axes[:2], (normal, agn), ("royalblue", "crimson")):
    for i in rng.choice(d["N"], 5, replace=False):
        ax.plot(wl, d["x"][i], lw=0.6, alpha=0.8)
    ax.set_title(f"{d['label']}: 5 random validation spectra ({flux_type})")
    ax.set_ylabel(flux_type)
    ax.grid(alpha=0.3)
for d, colour in ((normal, "royalblue"), (agn, "crimson")):
    axes[2].plot(wl, np.nanmedian(d["x"], axis=0), color=colour, label=f"{d['label']} median")
axes[2].axhline(ns["mean"], color="k", ls="--", lw=0.8, label="training mean")
axes[2].set_title("Median spectrum of each set")
axes[2].set_xlabel("Wavelength (Å)")
axes[2].set_ylabel(flux_type)
axes[2].legend()
axes[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"compare_agn_normal_{flux_type}.png", dpi=120)
print(f"\nSaved compare_agn_normal_{flux_type}.png")
