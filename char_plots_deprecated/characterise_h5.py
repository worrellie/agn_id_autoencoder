"""
characterise_h5.py

Characterise the scalar data in the compiled spectra HDF5 (all_spectra.h5).

Produces, for every 1-D numeric dataset (z, SNR, NORM factors, and all
catalogue params -- flux arrays and obj_id are skipped):

  PLOTS  (one file per parameter, written to ./char_plots/)
    * <param>_all.png   -- histogram of all splits pooled together
    * <param>_tv.png    -- train and validation histograms overlaid on one axes

  TABLES (printed to stdout and written to CSV)
    * stats_all.csv     -- range (min/max), mean, median for all splits pooled
    * stats_trainval.csv-- range, mean, median for train and validation separately

NaNs are dropped per-parameter before stats/plots (some params carry NaN for
galaxies where that quantity was unconstrained). Sentinel values (e.g.
TQUENCH == 99.0) are NOT stripped -- they are real stored values; they're
flagged in the printout instead.

Usage:
    python characterise_h5.py                # defaults to all_spectra.h5
    python characterise_h5.py my_file.h5
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

H5_PATH = sys.argv[1] if len(sys.argv) > 1 else "all_spectra.h5"
OUT_DIR = "char_plots"
SPLITS = ["train", "validation", "test"]

# datasets that are NOT per-galaxy scalar parameters -> excluded from characterisation
SKIP_DATASETS = {"raw_flux", "log_scale_flux", "obj_id", "skipped"}

# colours for the train/validation overlay
C_TRAIN = "#185FA5"
C_VALID = "#993C1D"


def discover_scalar_params(hf):
    """Return sorted names of 1-D numeric datasets shared across the splits."""
    # use the first present split as the schema reference
    ref = next((s for s in SPLITS if s in hf), None)
    if ref is None:
        raise SystemExit("no train/validation/test groups found in file")
    params = []
    for name, dset in hf[ref].items():
        if name in SKIP_DATASETS:
            continue
        if dset.ndim != 1:            # skips any 2-D array defensively
            continue
        if not np.issubdtype(dset.dtype, np.number):   # skips obj_id (bytes) etc.
            continue
        params.append(name)
    return sorted(params)


def load_param(hf, split, param):
    """Load one parameter for one split as a finite float array (NaN/inf dropped)."""
    if split not in hf or param not in hf[split]:
        return np.array([])
    v = np.asarray(hf[split][param][:], dtype=np.float64)
    return v[np.isfinite(v)]


def pooled(hf, param):
    """All splits concatenated for one parameter (finite only)."""
    return np.concatenate([load_param(hf, s, param) for s in SPLITS]) \
        if any(s in hf for s in SPLITS) else np.array([])


def stats_row(name, arr):
    """Range / mean / median summary for one array; handles empty gracefully."""
    if arr.size == 0:
        return {"param": name, "n": 0, "min": np.nan, "max": np.nan,
                "mean": np.nan, "median": np.nan, "std": np.nan}
    return {
        "param": name, "n": int(arr.size),
        "min": float(np.min(arr)),   "max": float(np.max(arr)),
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def print_table(title, rows):
    """Pretty fixed-width print of a list of stats-row dicts."""
    print(f"\n{title}\n" + "-" * len(title))
    hdr = f"{'param':<12}{'n':>8}{'min':>13}{'max':>13}{'mean':>13}{'median':>13}{'std':>13}"
    print(hdr)
    for r in rows:
        print(f"{r['param']:<12}{r['n']:>8}"
              f"{r['min']:>13.4g}{r['max']:>13.4g}"
              f"{r['mean']:>13.4g}{r['median']:>13.4g}{r['std']:>13.4g}")


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["param", "n", "min", "max", "mean", "median", "std"])
        w.writeheader()
        w.writerows(rows)


def positive_only(arr):
    """Values usable on a log x-axis: strictly > 0. Returns (kept, n_dropped)."""
    if arr.size == 0:
        return arr, 0
    keep = arr > 0
    return arr[keep], int(np.sum(~keep))


def _logx_note(ax, n_dropped, n_total):
    """Annotate a log-x plot with how many non-positive values were excluded."""
    if n_dropped:
        ax.text(0.99, 0.97, f"{n_dropped}/{n_total} non-positive excluded",
                ha="right", va="top", transform=ax.transAxes, fontsize=7, color="grey")


def plot_all(param, arr, out_path, logx=False):
    """Histogram of all splits pooled for one parameter. logx -> log10 x-axis."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

    n_total = arr.size
    if logx:
        arr, n_dropped = positive_only(arr)

    if arr.size:
        if logx:
            bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 60)
            ax.set_xscale("log")
        else:
            bins = 60
        ax.hist(arr, bins=bins, color="#3A6B35", alpha=0.85, edgecolor="none")
        ax.axvline(np.mean(arr),   color="k",       lw=1.2, ls="-",  label=f"mean {np.mean(arr):.3g}")
        ax.axvline(np.median(arr), color="darkred", lw=1.2, ls="--", label=f"median {np.median(arr):.3g}")
        ax.legend(fontsize=8)
        if logx:
            _logx_note(ax, n_dropped, n_total)
    else:
        msg = "no positive data for log axis" if logx else "no finite data"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(f"{param} (log x)" if logx else param)
    ax.set_ylabel("count")
    suffix = " [log x]" if logx else ""
    ax.set_title(f"{param} — all splits (n={n_total}){suffix}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_train_valid(param, tr, va, out_path, logx=False):
    """Train and validation histograms overlaid on one axes. logx -> log10 x-axis."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

    n_total = tr.size + va.size
    n_dropped = 0
    if logx:
        tr, d_tr = positive_only(tr)
        va, d_va = positive_only(va)
        n_dropped = d_tr + d_va

    # shared bin edges from the combined range so the two are directly comparable
    both = np.concatenate([tr, va]) if (tr.size or va.size) else np.array([])
    if both.size:
        if logx:
            bins = np.logspace(np.log10(both.min()), np.log10(both.max()), 60)
            ax.set_xscale("log")
        else:
            bins = np.linspace(both.min(), both.max(), 60)
        # density=True so unequal sample sizes (88k vs 11k) don't swamp the comparison
        if tr.size:
            ax.hist(tr, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                    color=C_TRAIN, label=f"train (n={tr.size})")
        if va.size:
            ax.hist(va, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                    color=C_VALID, label=f"validation (n={va.size})")
        ax.legend(fontsize=8)
        if logx:
            _logx_note(ax, n_dropped, n_total)
    else:
        msg = "no positive data for log axis" if logx else "no finite data"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(f"{param} (log x)" if logx else param)
    ax.set_ylabel("density")   # normalised, because train >> validation in count
    suffix = " [log x]" if logx else ""
    ax.set_title(f"{param} — train vs validation{suffix}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"characterising {H5_PATH}")

    with h5py.File(H5_PATH, "r") as hf:
        params = discover_scalar_params(hf)
        print(f"{len(params)} scalar parameters: {', '.join(params)}\n")

        all_rows = []          # stats_all.csv
        trainval_rows = []     # stats_trainval.csv

        for p in params:
            arr_all = pooled(hf, p)
            tr = load_param(hf, "train", p)
            va = load_param(hf, "validation", p)

            # --- plots: one file per parameter, per set, linear AND log-x ---
            plot_all(p, arr_all, os.path.join(OUT_DIR, f"{p}_all.png"))
            plot_all(p, arr_all, os.path.join(OUT_DIR, f"{p}_all_logx.png"), logx=True)
            plot_train_valid(p, tr, va, os.path.join(OUT_DIR, f"{p}_tv.png"))
            plot_train_valid(p, tr, va, os.path.join(OUT_DIR, f"{p}_tv_logx.png"), logx=True)

            # --- stats rows ---
            all_rows.append(stats_row(p, arr_all))
            trainval_rows.append(stats_row(f"{p} [train]", tr))
            trainval_rows.append(stats_row(f"{p} [valid]", va))

            # note any NaN-dropped fraction and TQUENCH-style sentinels
            raw_n = sum(hf[s][p].shape[0] for s in SPLITS if s in hf and p in hf[s])
            dropped = raw_n - arr_all.size
            if dropped:
                print(f"  note: {p} had {dropped}/{raw_n} non-finite values dropped")

        # --- tables: print + CSV ---
        print_table("RANGE / MEAN / MEDIAN — all splits pooled", all_rows)
        print_table("RANGE / MEAN / MEDIAN — train and validation separately", trainval_rows)

        write_csv("stats_all.csv", all_rows)
        write_csv("stats_trainval.csv", trainval_rows)

    print(f"\nwrote {4*len(params)} plots to {OUT_DIR}/ "
          f"(_all, _all_logx, _tv, _tv_logx per param)")
    print("wrote stats_all.csv and stats_trainval.csv")


if __name__ == "__main__":
    main()
