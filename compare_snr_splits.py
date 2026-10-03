#!/usr/bin/env python3
"""
Compare SNR distributions of sources in each split of two HDF5 spectra files.
Figures (one row per split, one column per dataset):
    snr_comparison.png         SNR_MED | SNR_MEAN   (log x-axis)
    snr_comparison_linear.png  SNR_MED | SNR_MEAN   (linear x-axis, up to the 99th percentile)
    noise_norm_comparison.png  NOISE | NORM_CMN | NORM_CMD | NORM_MED

Expected layout (as produced by the SAE HDF5 compilation):
    file.h5
    ├── training/     (optional; may exist in only one file)
    ├── validation/
    │   ├── SNR_MED   (N,)
    │   ├── SNR_MEAN  (N,)
    │   └── obj_id    (N,) bytes
    └── test/

Splits present in both files are compared (summary stats, KS test, and a
matched-source check via obj_id). Splits present in only one file are still
summarised and plotted on their own.

Figures are saved next to file_a (the first .h5) unless --outdir is given.

Usage:
    python compare_snr_splits.py all_agn_float32.h5 other.h5 --labels AGN other   # AGN red, other blue
    python compare_snr_splits.py a.h5 b.h5 --keys NOISE EBV --out x.png   # custom figure
"""
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SPLIT_ORDER = ["training", "train", "validation", "val", "test"]
SNR_THRESHOLDS = [0, 1, 3, 5, 10]

# default figures: output file -> (datasets, one column each left to right; log x-axis?)
FIGURES = {
    "snr_comparison.png": (["SNR_MED", "SNR_MEAN"], True),
    "snr_comparison_linear.png": (["SNR_MED", "SNR_MEAN"], False),
    "noise_norm_comparison.png": (["NOISE", "NORM_CMN", "NORM_CMD", "NORM_MED"], True),
}


def splits_with(f, key):
    return [k for k in f.keys() if isinstance(f[k], h5py.Group) and key in f[k]]


def load(f, split, key):
    """Return (snr, obj_id) with non-finite SNR removed, plus count dropped."""
    g = f[split]
    snr = g[key][()].astype(np.float64)
    ids = g["obj_id"][()] if "obj_id" in g else None
    ok = np.isfinite(snr)
    return snr[ok], (ids[ok] if ids is not None else None), int((~ok).sum())


def summary_row(split, label, x):
    p5, p16, p50, p84, p95 = np.percentile(x, [5, 16, 50, 84, 95])
    return (f"{split:<11}{str(label)[:13]:<14}{x.size:>7d}{x.mean():>9.3g}{p50:>9.3g}"
            f"{p5:>9.3g}{p16:>9.3g}{p84:>9.3g}{p95:>9.3g}{(x <= 0).sum():>7d}")


def matched_check(a, ida, b, idb):
    """If the same sources appear in both files, compare their SNR one-to-one."""
    if ida is None or idb is None:
        return None
    common, ia, ib = np.intersect1d(ida, idb, return_indices=True)
    if common.size == 0:
        return "no obj_id overlap (different sources)"
    d = b[ib] - a[ia]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = b[ib] / a[ia]
    r = r[np.isfinite(r) & (a[ia] > 0)]
    return (f"{common.size} matched sources | median Δ={np.median(d):.3g} | "
            f"median ratio B/A={np.median(r):.3g} | identical={np.sum(d == 0)}")


def plot_hist(ax, title, series, bins_n, colours, log=True, linear_max_pct=99):
    """series: list of (label, values).

    log=True:  log-spaced bins over the positive values (values ≤0 can't be shown).
    log=False: linear bins from the minimum up to the `linear_max_pct` percentile of the
               combined data, so the long upper tail doesn't squash the bulk; values
               beyond the x-limit are counted in the legend rather than silently dropped.
    """
    allx = np.concatenate([x for _, x in series])
    if log:
        pos = allx[allx > 0]
        bins = np.geomspace(pos.min(), pos.max(), bins_n + 1)
    else:
        bins = np.linspace(allx.min(), np.percentile(allx, linear_max_pct), bins_n + 1)
    for lab, x in series:
        shown = (x >= bins[0]) & (x <= bins[-1])
        n_out = int((~shown).sum())
        if n_out == 0:
            note = ""
        elif log:
            note = f", {n_out} ≤0 not shown"
        else:
            note = f", {n_out} > {bins[-1]:.3g} not shown"
        # fraction of all sources per bin (density=True would distort log-spaced bins)
        ax.hist(x[shown], bins=bins, weights=np.full(shown.sum(), 1 / x.size), histtype="step",
                lw=1.5, color=colours[lab],
                label=f"{lab} (N={x.size}, med={np.median(x):.2g}{note})")
    ax.set(title=title, ylabel="fraction per bin", xscale="log" if log else "linear")
    ax.legend(fontsize=7)


def run_key(fa, fb, key, la, lb):
    """Print stats for one dataset; return {split: [(label, values), ...]}."""
    sa, sb = splits_with(fa, key), splits_with(fb, key)
    if not sa and not sb:
        print(f"'{key}' not found in either file, skipping.")
        return {}
    print(f"\n=== {key} ===")
    print(f"{'split':<11}{'file':<14}{'N':>7}{'mean':>9}{'median':>9}{'p5':>9}"
          f"{'p16':>9}{'p84':>9}{'p95':>9}{'n≤0':>7}")
    out = {}
    for s in sorted(set(sa) | set(sb), key=split_order):
        series, loaded = [], {}
        for f, lab, present in ((fa, la, s in sa), (fb, lb, s in sb)):
            if present:
                x, ids, dropped = load(f, s, key)
                if dropped:
                    print(f"  [{s}/{lab}] dropped {dropped} non-finite values")
                loaded[lab] = (x, ids)
                series.append((lab, x))
                print(summary_row(s, lab, x))
        if len(loaded) == 2:
            (a, ida), (b, idb) = loaded[la], loaded[lb]
            ks = stats.ks_2samp(a, b)
            print(f"{'':<11}KS D={ks.statistic:.4f}  p={ks.pvalue:.3g}")
            m = matched_check(a, ida, b, idb)
            if m:
                print(f"{'':<11}{m}")
        else:
            print(f"{'':<11}(only in {series[0][0]}, not compared)")
        out[s] = series
    return out


def print_thresholds(key, by_split, la, lb):
    """Counts (and %) of sources above each SNR threshold, per split and in total, per file."""
    cols = "".join(f"{'> ' + str(t):>16}" for t in SNR_THRESHOLDS)
    print(f"\n--- {key}: sources above SNR threshold, count (% of file's split) ---")
    print(f"{'split':<11}{'file':<14}{'N':>7}{cols}")
    totals = {la: [], lb: []}
    for s, series in by_split.items():
        for lab, x in series:
            totals[lab].append(x)
            print(f"{s:<11}{str(lab)[:13]:<14}{x.size:>7d}" + "".join(
                f"{(n := int((x > t).sum())):>8d} ({100 * n / x.size:4.1f}%)" for t in SNR_THRESHOLDS))
    for lab, xs in totals.items():
        if xs:
            x = np.concatenate(xs)
            print(f"{'ALL':<11}{str(lab)[:13]:<14}{x.size:>7d}" + "".join(
                f"{(n := int((x > t).sum())):>8d} ({100 * n / x.size:4.1f}%)" for t in SNR_THRESHOLDS))


def split_order(s):
    return (SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 99, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--keys", nargs="+", default=None,
                    help="make a single custom figure from these datasets instead of the defaults")
    ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--colours", nargs=2, default=["tab:red", "tab:blue"],
                    help="line colours for file_a and file_b (default: red, blue)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--out", default="custom_comparison.png", help="output name when using --keys")
    ap.add_argument("--linear", action="store_true", help="linear x-axis for a --keys figure")
    ap.add_argument("--outdir", default=None,
                    help="where to save figures (default: the folder containing file_a)")
    args = ap.parse_args()

    la, lb = args.labels or (os.path.basename(args.file_a), os.path.basename(args.file_b))
    colours = {la: args.colours[0], lb: args.colours[1]}
    figures = {args.out: (args.keys, not args.linear)} if args.keys else FIGURES
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.file_a))
    os.makedirs(outdir, exist_ok=True)
    figures = {os.path.join(outdir, out): v for out, v in figures.items()}
    cache = {}   # stats are printed once per dataset, even if it appears in several figures
    with h5py.File(args.file_a, "r") as fa, h5py.File(args.file_b, "r") as fb:
        for out, (keys, log) in figures.items():
            for k in keys:
                if k not in cache:
                    cache[k] = run_key(fa, fb, k, la, lb)
            make_figure({k: cache[k] for k in keys}, colours, args.bins, out, log)
        for k, by_split in cache.items():
            if k.startswith("SNR") and by_split:
                print_thresholds(k, by_split, la, lb)


def make_figure(results, colours, bins_n, out, log=True):
    """Rows = splits, columns = datasets. results: {key: {split: series}}."""
    results = {k: v for k, v in results.items() if v}
    if not results:
        return
    splits = sorted({s for v in results.values() for s in v}, key=split_order)
    keys = list(results)
    fig, axes = plt.subplots(len(splits), len(keys), figsize=(5.5 * len(keys), 3.3 * len(splits)),
                             squeeze=False)
    for i, s in enumerate(splits):
        for j, k in enumerate(keys):
            if s in results[k]:
                plot_hist(axes[i, j], f"{s}: {k}", results[k][s], bins_n, colours, log=log)
            else:
                axes[i, j].set_visible(False)
    for j, k in enumerate(keys):
        axes[-1, j].set_xlabel(k + ("" if k.startswith("SNR") else "  [flux units]"))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
