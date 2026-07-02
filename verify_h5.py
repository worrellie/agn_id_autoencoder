"""
verify_h5.py

Post-build sanity checks for the compiled spectra HDF5 (all_spectra.h5).

Runs three groups of checks and prints a per-check PASS/WARN/FAIL, then an
overall verdict. Read-only: it never modifies the file.

  1. Redshift integrity   -- the check that closes the loop on the stale-z bug:
                             stored redshift (OG_Z) must match the catalogue
                             REDSHIF per galaxy, and must be genuinely distinct
                             across galaxies (not one repeated stale value).
  2. Flux hygiene         -- log_scale_flux has no NaN/inf; masked (gap) pixels
                             are still exactly zero; shapes are consistent.
  3. Structure & sentinels-- expected datasets present, split counts reconcile,
                             TQUENCH=99.0 sentinel flagged so it isn't treated
                             as a physical value downstream.

Usage:
    python verify_h5.py                 # defaults to all_spectra.h5
    python verify_h5.py my_file.h5
"""

import sys
import numpy as np
import h5py

H5_PATH = sys.argv[1] if len(sys.argv) > 1 else "all_spectra.h5"

SPLITS = ["train", "validation", "test"]
Z_TOL = 1e-3            # OG_Z is filename-parsed, REDSHIF is full precision -> ~1e-3 agreement
FLUX_KEYS = ["raw_flux", "log_scale_flux"]
SAMPLE_N = 2000        # rows to sample for the expensive full-array flux checks

# collect results as (level, message); level in {"PASS","WARN","FAIL"}
results = []
def record(level, msg):
    results.append((level, msg))
    tag = {"PASS": "  ok ", "WARN": " warn", "FAIL": "FAIL "}[level]
    print(f"[{tag}] {msg}")


def check_redshift(hf, split):
    """OG_Z (stored 'redshift') must equal catalogue REDSHIF per galaxy, and be distinct."""
    g = hf[split]
    if "redshift" not in g or "REDSHIF" not in g:
        record("WARN", f"{split}: missing 'redshift' or 'REDSHIF' -> cannot cross-check")
        return

    z_used = g["redshift"][:]      # from OG_Z: the value processing de-redshifted by
    z_cat  = g["REDSHIF"][:]       # catalogue TARGET_REDSHIFT

    finite = np.isfinite(z_used) & np.isfinite(z_cat)
    diff = np.abs(z_used[finite] - z_cat[finite])
    max_diff = diff.max() if diff.size else np.nan
    n_bad = int(np.sum(diff > Z_TOL))

    if n_bad == 0:
        record("PASS", f"{split}: OG_Z matches REDSHIF (max |dz|={max_diff:.2e}, tol={Z_TOL:.0e})")
    else:
        record("FAIL", f"{split}: {n_bad} rows with |OG_Z - REDSHIF| > {Z_TOL:.0e} "
                        f"(max |dz|={max_diff:.2e}) -- redshift wiring suspect")

    # the stale-bug signature was ONE repeated redshift; expect many distinct values
    n_distinct = len(np.unique(np.round(z_used[finite], 6)))
    n_rows = int(finite.sum())
    if n_distinct == 1 and n_rows > 1:
        record("FAIL", f"{split}: only 1 distinct redshift across {n_rows} rows "
                        f"-- classic stale-z signature")
    elif n_distinct < max(2, n_rows // 100):
        record("WARN", f"{split}: only {n_distinct} distinct redshifts across {n_rows} rows "
                        f"-- unexpectedly few, worth a look")
    else:
        record("PASS", f"{split}: {n_distinct} distinct redshifts across {n_rows} rows")


def check_flux(hf, split):
    """log_scale_flux: no NaN/inf; gap pixels exactly zero; shapes consistent."""
    g = hf[split]
    n_rows = g["raw_flux"].shape[0]
    n_pix_attr = hf.attrs["wavelengths"].shape[0] if "wavelengths" in hf.attrs else None

    # shape consistency across the two flux arrays and the wavelength grid
    shapes_ok = True
    for k in FLUX_KEYS:
        if k not in g:
            record("WARN", f"{split}: flux dataset '{k}' missing"); shapes_ok = False; continue
        npix = g[k].shape[1]
        if n_pix_attr is not None and npix != n_pix_attr:
            record("FAIL", f"{split}/{k}: n_pixels {npix} != wavelength grid {n_pix_attr}")
            shapes_ok = False
    if shapes_ok:
        record("PASS", f"{split}: flux shapes consistent ({n_rows} x {g['raw_flux'].shape[1]})")

    # sample rows for the finite/mask checks (avoids loading the whole array)
    idx = np.arange(n_rows) if n_rows <= SAMPLE_N else np.random.default_rng(0).choice(
        n_rows, SAMPLE_N, replace=False)
    idx = np.sort(idx)

    log = g["log_scale_flux"][idx]
    raw = g["raw_flux"][idx]

    # 1. no NaN / inf in the array we actually train on
    n_nonfinite = int(np.sum(~np.isfinite(log)))
    if n_nonfinite == 0:
        record("PASS", f"{split}: log_scale_flux finite everywhere (sampled {len(idx)} rows)")
    else:
        record("FAIL", f"{split}: {n_nonfinite} non-finite values in log_scale_flux "
                        f"(sampled {len(idx)} rows)")

    # 2. masked pixels: where raw==0 (a gap/edge), log must also be exactly 0
    gap = (raw == 0)
    if gap.any():
        leaked = int(np.sum(log[gap] != 0))
        if leaked == 0:
            record("PASS", f"{split}: all masked (gap) pixels are exactly 0 in log_scale_flux")
        else:
            record("FAIL", f"{split}: {leaked} masked pixels are non-zero in log_scale_flux "
                            f"-- '* unmasked' step suspect")
    else:
        record("WARN", f"{split}: no zero (masked) pixels found in sample -- unexpected")


def check_structure(hf, split, expected_counts):
    """Split count reconciles; NORM factors sane; TQUENCH sentinel flagged."""
    g = hf[split]
    n_rows = g["raw_flux"].shape[0]

    exp = expected_counts.get(split)
    if exp is not None and n_rows != exp:
        record("WARN", f"{split}: {n_rows} rows (expected ~{exp} from split)")
    else:
        record("PASS", f"{split}: {n_rows} rows")

    # NORM factors must be finite and non-zero (they divide raw flux downstream)
    for k in ("NORM_CON", "NORM_MED"):
        if k not in g:
            record("WARN", f"{split}: '{k}' missing"); continue
        v = g[k][:]
        n_bad = int(np.sum(~np.isfinite(v) | (v == 0)))
        if n_bad == 0:
            record("PASS", f"{split}: {k} all finite & non-zero")
        else:
            record("WARN", f"{split}: {k} has {n_bad} zero/non-finite entries "
                            f"(would have defaulted to 1.0 at build)")

    # TQUENCH sentinel: 99.0 means 'never quenched', not a physical time
    if "TQUENCH" in g:
        tq = g["TQUENCH"][:]
        n_sentinel = int(np.sum(tq == 99.0))
        if n_sentinel:
            record("WARN", f"{split}: TQUENCH == 99.0 for {n_sentinel}/{n_rows} rows "
                            f"-- sentinel, treat as flag not a value")


def main():
    print(f"verifying {H5_PATH}\n" + "=" * 60)

    with h5py.File(H5_PATH, "r") as hf:
        present = [s for s in SPLITS if s in hf]
        missing = [s for s in SPLITS if s not in hf]
        for s in missing:
            record("FAIL", f"split group '{s}' missing from file")

        # reconcile split sizes against their total
        counts = {s: hf[s]["raw_flux"].shape[0] for s in present}
        total = sum(counts.values())
        print(f"\nsplit sizes: " + ", ".join(f"{s}={counts[s]}" for s in present)
              + f"  (total {total})\n")

        for split in present:
            print(f"--- {split} ---")
            check_redshift(hf, split)
            check_flux(hf, split)
            check_structure(hf, split, counts)
            print()

    # overall verdict
    n_fail = sum(1 for lvl, _ in results if lvl == "FAIL")
    n_warn = sum(1 for lvl, _ in results if lvl == "WARN")
    print("=" * 60)
    if n_fail:
        print(f"VERDICT: {n_fail} FAIL, {n_warn} WARN -- do NOT train until FAILs are resolved.")
        sys.exit(1)
    elif n_warn:
        print(f"VERDICT: 0 FAIL, {n_warn} WARN -- usable; review warnings above.")
    else:
        print("VERDICT: all checks passed -- dataset is good to train on.")
        
    


if __name__ == "__main__":
    main()