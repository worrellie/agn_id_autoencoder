"""Exposure-time and H-band magnitude distributions.

Works on both files:
  - normal galaxies (train/validation/test, has HMAG): exposure time + all HMAG figures
  - AGN (validation/test, no HMAG): exposure time figure only

Exposure time is read from each spectrum's source filename (obj_id),
e.g. cosmos_bagpipes_202598_2h_z1.2546_... -> 2 h, which works for both samples.
H-band magnitude is the HMAG dataset, taken from the Bagpipes/COSMOS catalogue.

Figures are prefixed with the file's name, so galaxy and AGN outputs don't overwrite each other.

Usage:
    uv run python plot_exptime_hmag.py all_spectra_float32_v3.h5
    uv run python plot_exptime_hmag.py all_agn_float32.h5
"""
import pathlib
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

EXP_RE = re.compile(r"_(\d+(?:\.\d+)?)h_")
SPLIT_COLOURS = {"train": "royalblue", "validation": "darkorange", "test": "seagreen"}
HMAG_SENTINEL = -99.0   # catalogue placeholder for "no measurement"


def hmag_masks(hmag):
    """Return (valid, sentinel) boolean masks. Sentinel and NaN values are excluded from plots."""
    sentinel = np.isclose(hmag, HMAG_SENTINEL)
    return np.isfinite(hmag) & ~sentinel, sentinel


def exposure_hours(obj_ids):
    """Parse exposure time in hours from each filename; NaN if not found."""
    out = np.full(len(obj_ids), np.nan)
    for i, raw in enumerate(obj_ids):
        m = EXP_RE.search(raw.decode() if isinstance(raw, bytes) else raw)
        if m:
            out[i] = float(m.group(1))
    return out


def load(path):
    """Returns ({split: {"exp", "hmag"}}, has_hmag). hmag is None for files without HMAG."""
    data = {}
    with h5py.File(path, "r") as hf:
        splits = [s for s in ("train", "validation", "test") if s in hf]
        if not splits:
            sys.exit(f"{path}: no train/validation/test groups found")
        has_hmag = all("HMAG" in hf[s] for s in splits)
        for s in splits:
            data[s] = {"exp": exposure_hours(hf[s]["obj_id"][:]),
                       "hmag": hf[s]["HMAG"][:].astype(float) if has_hmag else None}
    return data, has_hmag


def plot_exposure(data, sample="normal galaxies"):
    exps = np.unique(np.concatenate([d["exp"][np.isfinite(d["exp"])] for d in data.values()]))
    x = np.arange(len(exps))
    width = 0.8 / len(data)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (split, d) in enumerate(data.items()):
        counts = [np.sum(d["exp"] == e) for e in exps]
        bars = ax.bar(x + (k - (len(data) - 1) / 2) * width, counts, width,
                      color=SPLIT_COLOURS.get(split), label=f"{split} (N={len(d['exp'])})")
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x, [f"{e:g} h" for e in exps])
    ax.set_xlabel("Exposure time")
    ax.set_ylabel("N")
    ax.set_title(f"Exposure time distribution ({sample})")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)   # headroom so the legend clears the bars
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_hmag(data, log_n=False):
    valid = {split: d["hmag"][hmag_masks(d["hmag"])[0]] for split, d in data.items()}
    n_sentinel = {split: int(hmag_masks(d["hmag"])[1].sum()) for split, d in data.items()}
    all_h = np.concatenate(list(valid.values()))
    bins = np.arange(np.floor(all_h.min() * 4) / 4, np.ceil(all_h.max() * 4) / 4 + 0.25, 0.25)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, h in valid.items():
        ax.hist(h, bins=bins, histtype="step", lw=1.5, color=SPLIT_COLOURS.get(split),
                label=f"{split} (N={len(h)})")
    ax.set_xlabel("H-band magnitude")
    if log_n:
        ax.set_yscale("log")   # log N: makes the sparse bright and faint tails visible
    ax.set_ylabel("N (log scale)" if log_n else "N")
    ax.set_title("H-band magnitude distribution (normal galaxies)")
    ax.legend()
    note = (f"Not plotted: {sum(n_sentinel.values())} with HMAG = {HMAG_SENTINEL:g} ("
            + ", ".join(f"{s}: {n}" for s, n in n_sentinel.items()) + ")")
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))   # leave room below the axes for the note
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
    return fig


def plot_hmag_by_exposure(data, exposures=(2, 8), colours=("purple", "goldenrod")):
    """H-band magnitude distribution for selected exposure times, all splits combined."""
    exp = np.concatenate([d["exp"] for d in data.values()])
    hmag = np.concatenate([d["hmag"] for d in data.values()])
    valid, sentinel = hmag_masks(hmag)

    selected = {e: hmag[valid & (exp == e)] for e in exposures}
    n_sentinel = {e: int((sentinel & (exp == e)).sum()) for e in exposures}
    all_h = np.concatenate(list(selected.values()))
    bins = np.arange(np.floor(all_h.min() * 4) / 4, np.ceil(all_h.max() * 4) / 4 + 0.25, 0.25)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (e, h), colour in zip(selected.items(), colours):
        ax.hist(h, bins=bins, histtype="step", lw=1.5, color=colour,
                label=f"{e:g} h (N={len(h)}, median H={np.median(h):.2f})")
    ax.set_xlabel("H-band magnitude")
    ax.set_ylabel("N")
    ax.set_title("H-band magnitude by exposure time (normal galaxies, all splits)")
    ax.legend()
    ax.grid(alpha=0.3)
    note = (f"Not plotted: {sum(n_sentinel.values())} with HMAG = {HMAG_SENTINEL:g} ("
            + ", ".join(f"{e:g} h: {n}" for e, n in n_sentinel.items()) + ")")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
    return fig


def summarise(data, has_hmag):
    header = f"{'split':<11}{'N':>8}{'no exp':>8}"
    if has_hmag:
        header += f"{'HMAG NaN':>10}{'HMAG -99':>10}{'HMAG range (valid)':>21}"
    print(header + "   exposure counts")
    for split, d in data.items():
        e = d["exp"][np.isfinite(d["exp"])]
        counts = ", ".join(f"{v:g}h: {c}" for v, c in zip(*np.unique(e, return_counts=True)))
        line = f"{split:<11}{len(d['exp']):>8}{np.isnan(d['exp']).sum():>8}"
        if has_hmag:
            valid, sentinel = hmag_masks(d["hmag"])
            h = d["hmag"][valid]
            line += (f"{np.isnan(d['hmag']).sum():>10}{sentinel.sum():>10}"
                     f"{f'{h.min():.2f} – {h.max():.2f}':>21}")
        print(line + f"   {counts}")


if __name__ == "__main__":
    path = sys.argv[1]
    data, has_hmag = load(path)
    sample = "normal galaxies" if has_hmag else "AGN"
    prefix = pathlib.Path(path).stem
    summarise(data, has_hmag)

    figures = [("exposure_time", plot_exposure(data, sample))]
    if has_hmag:
        figures += [("hmag", plot_hmag(data)),
                    ("hmag_logN", plot_hmag(data, log_n=True)),
                    ("hmag_by_exposure_2h_8h", plot_hmag_by_exposure(data, exposures=(2, 8)))]
    else:
        print(f"{path}: no HMAG — exposure-time figure only")

    for name, fig in figures:
        for ext in ("png", "pdf"):
            fig.savefig(f"{prefix}_{name}_distribution.{ext}", dpi=150)
        print(f"saved {prefix}_{name}_distribution.png/.pdf")