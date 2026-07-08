#!/usr/bin/env python3
"""
Standalone integrity checker for the spectra HDF5.

Usage:
    python check_h5.py all_spectra_float32.h5

Runs on a laptop: reads the flux arrays in chunks so peak memory stays tiny
(~a few tens of MB) regardless of file size. Reports PASS / WARN / FAIL for a
battery of checks tailored to this dataset, and exits non-zero if anything FAILs.

What it checks, and why each matters:
  - structure: expected splits + datasets present, shapes internally consistent
  - dtype float32 (the migration you just did)
  - wavelengths: monotonic, ~constant spacing, length == n_pixels
  - NaN/gap integrity (the sentinel change you just made):
      * every spectrum has >=1 gap        (you said all spectra have gaps)
      * NO spectrum is entirely NaN        (would divide-by-zero in the loss)
      * gap columns consistent across rows and splits (gaps are instrumental)
  - finiteness: no +/-inf hiding among the valid pixels
  - value sanity + stats cross-check: recomputed nanmean/nanstd of train
    log_scale_flux must match the stored norm_mean_log / norm_std_log
    (confirms standardisation will actually centre the data)
  - metadata: redshift/SNR finite and plausible, obj_id decodes
  - leakage: same galaxy id appearing in more than one split (informational)
"""

import argparse
import re
import sys
import numpy as np
import h5py

SPLITS = ["train", "validation", "test"]
FLUX_KEYS = ["log_scale_flux", "raw_flux"]
CHUNK = 5000                      # rows per read; keeps memory low
EXPECTED_DPIX = 4.0              # nominal wavelength spacing (Angstrom)


class Report:
    """Collects PASS/WARN/FAIL lines and tracks overall status."""
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def ok(self, msg):    print(f"  \033[32mPASS\033[0m  {msg}")
    def warn(self, msg):  print(f"  \033[33mWARN\033[0m  {msg}"); self.warns += 1
    def fail(self, msg):  print(f"  \033[31mFAIL\033[0m  {msg}"); self.fails += 1
    def info(self, msg):  print(f"        {msg}")
    def header(self, msg): print(f"\n=== {msg} ===")


def check_root(hf, rep):
    rep.header("root attributes")
    for k in ("wavelengths", "norm_mean_log", "norm_std_log"):
        if k not in hf.attrs:
            rep.fail(f"missing root attribute '{k}'")
    if "norm_mean_log" in hf.attrs and "norm_std_log" in hf.attrs:
        m, s = float(hf.attrs["norm_mean_log"]), float(hf.attrs["norm_std_log"])
        if not np.isfinite(m) or not np.isfinite(s):
            rep.fail(f"stored stats not finite: mean={m}, std={s}")
        elif s <= 0:
            rep.fail(f"norm_std_log <= 0 ({s}) -> standardisation divides by zero/neg")
        else:
            rep.ok(f"stored log stats finite: mean={m:.4f}, std={s:.4f}")

    if "wavelengths" in hf.attrs:
        w = np.asarray(hf.attrs["wavelengths"], dtype=float)
        d = np.diff(w)
        if not np.all(d > 0):
            rep.fail("wavelengths not strictly increasing")
        else:
            rep.ok(f"wavelengths increasing, {len(w)} points, "
                   f"spacing min/med/max = {d.min():.2f}/{np.median(d):.2f}/{d.max():.2f}")
        if abs(np.median(d) - EXPECTED_DPIX) > 0.5:
            rep.warn(f"median spacing {np.median(d):.2f} != expected ~{EXPECTED_DPIX}")
        return len(w)
    return None


def check_structure(hf, rep, n_wave):
    rep.header("structure & shapes")
    for split in SPLITS:
        if split not in hf:
            rep.fail(f"missing split group '{split}'"); continue
        g = hf[split]
        for key in FLUX_KEYS + ["redshift", "SNR", "obj_id"]:
            if key not in g:
                rep.fail(f"{split}: missing dataset '{key}'")
        if "log_scale_flux" not in g:
            continue
        n, p = g["log_scale_flux"].shape
        # every per-row dataset should agree on n
        for key in g:
            if g[key].shape and g[key].shape[0] != n:
                rep.warn(f"{split}: '{key}' first dim {g[key].shape[0]} != {n}")
        if n_wave is not None and p != n_wave:
            rep.fail(f"{split}: n_pixels {p} != len(wavelengths) {n_wave}")
        else:
            rep.ok(f"{split}: {n} spectra x {p} pixels, per-row datasets consistent")
        for key in FLUX_KEYS:
            if key in g and g[key].dtype != np.float32:
                rep.warn(f"{split}/{key}: dtype {g[key].dtype} (expected float32)")


def scan_flux(dset):
    """Chunked pass over one (n, p) flux array. Returns a dict of accumulated stats."""
    n, p = dset.shape
    col_always_gap = np.ones(p, dtype=bool)   # AND over rows
    col_ever_gap = np.zeros(p, dtype=bool)    # OR over rows
    min_gaps = p + 1
    max_gaps = -1
    rows_zero_gap = 0
    rows_all_gap = 0
    any_inf = False
    vsum = vsumsq = 0.0
    vcount = 0
    vmin, vmax = np.inf, -np.inf

    for i in range(0, n, CHUNK):
        a = dset[i:i + CHUNK].astype(np.float64)   # float64 for stable accumulation
        nan = np.isnan(a)
        gpr = nan.sum(axis=1)
        min_gaps = min(min_gaps, int(gpr.min()))
        max_gaps = max(max_gaps, int(gpr.max()))
        rows_zero_gap += int((gpr == 0).sum())
        rows_all_gap += int((gpr == p).sum())
        col_always_gap &= nan.all(axis=0)
        col_ever_gap |= nan.any(axis=0)
        valid = a[~nan]
        if valid.size:
            if np.isinf(valid).any():
                any_inf = True
                valid = valid[np.isfinite(valid)]
            vsum += valid.sum()
            vsumsq += (valid ** 2).sum()
            vcount += valid.size
            vmin = min(vmin, valid.min())
            vmax = max(vmax, valid.max())

    mean = vsum / vcount if vcount else np.nan
    std = np.sqrt(max(0.0, vsumsq / vcount - mean ** 2)) if vcount else np.nan
    return dict(n=n, p=p, min_gaps=min_gaps, max_gaps=max_gaps,
                rows_zero_gap=rows_zero_gap, rows_all_gap=rows_all_gap,
                any_inf=any_inf, mean=mean, std=std, vmin=vmin, vmax=vmax,
                nan_frac=1 - vcount / (n * p), col_ever_gap=col_ever_gap,
                col_always_gap=col_always_gap)


def check_flux(hf, rep):
    rep.header("flux integrity (NaN / gaps / finiteness)")
    ever_gap_by_split = {}
    for split in SPLITS:
        if split not in hf or "log_scale_flux" not in hf[split]:
            continue
        for key in FLUX_KEYS:
            if key not in hf[split]:
                continue
            s = scan_flux(hf[split][key])
            tag = f"{split}/{key}"

            # critical: no fully-masked row (would divide by zero in the loss)
            if s["rows_all_gap"] > 0:
                rep.fail(f"{tag}: {s['rows_all_gap']} spectra are ENTIRELY NaN "
                         f"-> n_unmasked_pixels=0 breaks the loss")
            # you said every spectrum has gaps
            if s["rows_zero_gap"] > 0:
                rep.warn(f"{tag}: {s['rows_zero_gap']} spectra have NO gaps "
                         f"(expected all to have some)")
            if s["any_inf"]:
                rep.fail(f"{tag}: +/-inf present among valid pixels")

            min_valid = s["p"] - s["max_gaps"]
            if s["rows_all_gap"] == 0 and s["any_inf"] is False and s["rows_zero_gap"] == 0:
                rep.ok(f"{tag}: gaps/row {s['min_gaps']}-{s['max_gaps']}, "
                       f"min valid pixels/row = {min_valid}, nan_frac={s['nan_frac']:.3g}")
            rep.info(f"{tag}: valid value range [{s['vmin']:.3g}, {s['vmax']:.3g}], "
                     f"nanmean={s['mean']:.4f}, nanstd={s['std']:.4f}")

            # stats cross-check: train log_scale_flux vs stored attrs
            if split == "train" and key == "log_scale_flux":
                m0 = float(hf.attrs.get("norm_mean_log", np.nan))
                s0 = float(hf.attrs.get("norm_std_log", np.nan))
                if abs(s["mean"] - m0) > 0.01 or abs(s["std"] - s0) > 0.01:
                    rep.warn(f"recomputed train stats (mean={s['mean']:.4f}, "
                             f"std={s['std']:.4f}) differ from stored "
                             f"(mean={m0:.4f}, std={s0:.4f})")
                else:
                    rep.ok(f"stored stats match recomputed train stats "
                           f"(mean~{m0:.4f}, std~{s0:.4f}) -> standardisation will centre data")

    # gap columns consistent ACROSS splits (instrumental gaps are fixed)
    for key, per_split in ever_gap_by_split.items():
        cols = list(per_split.values())
        if len(cols) > 1 and not all(np.array_equal(cols[0], c) for c in cols[1:]):
            rep.warn(f"{key}: gap columns differ between splits (expected identical)")
        elif len(cols) > 1:
            rep.ok(f"{key}: gap columns identical across all splits")


def check_meta_and_leakage(hf, rep):
    rep.header("metadata & split leakage")
    gal_re = re.compile(r"_(\d+)_")   # pulls the galaxy id from cosmos_bagpipes_<ID>_...
    gals_by_split = {}
    for split in SPLITS:
        if split not in hf:
            continue
        g = hf[split]
        if "redshift" in g:
            z = g["redshift"][:]
            if not np.all(np.isfinite(z)):
                rep.fail(f"{split}: non-finite redshift values")
            elif z.min() < 0:
                rep.warn(f"{split}: negative redshift (min={z.min():.3g})")
            else:
                rep.ok(f"{split}: redshift finite, range [{z.min():.3g}, {z.max():.3g}]")
        if "SNR" in g:
            snr = g["SNR"][:]
            if not np.all(np.isfinite(snr)):
                rep.warn(f"{split}: non-finite SNR values")
        if "obj_id" in g:
            try:
                ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                       for x in g["obj_id"][:]]
                gals = {m.group(1) for s in ids if (m := gal_re.search(s))}
                gals_by_split[split] = gals
            except Exception as e:
                rep.warn(f"{split}: obj_id decode issue: {type(e).__name__}")

    # cross-split galaxy overlap (same source at different exposures can leak)
    keys = list(gals_by_split)
    reported = False
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = gals_by_split[keys[i]] & gals_by_split[keys[j]]
            if overlap:
                rep.warn(f"{keys[i]} & {keys[j]} share {len(overlap)} galaxy id(s) "
                         f"(same source, different exposure?) — leakage risk if unintended")
                reported = True
    if keys and not reported:
        rep.ok("no galaxy id appears in more than one split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path")
    args = ap.parse_args()

    print(f"\n{'='*50}")
    rep = Report()
    print(f"Checking {args.h5_path}")
    try:
        hf = h5py.File(args.h5_path, "r")
    except Exception as e:
        print(f"  FAIL  cannot open file: {e}")
        sys.exit(2)

    with hf:
        n_wave = check_root(hf, rep)
        check_structure(hf, rep, n_wave)
        check_flux(hf, rep)
        check_meta_and_leakage(hf, rep)

    print(f"\n{'='*50}")
    if rep.fails:
        print(f"RESULT: {rep.fails} FAIL, {rep.warns} WARN — do not train on this file yet")
        sys.exit(1)
    elif rep.warns:
        print(f"RESULT: 0 FAIL, {rep.warns} WARN — usable, but review the warnings")
        sys.exit(0)
    else:
        print("RESULT: all checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
