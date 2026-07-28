"""
verify_med_h5.py — sanity-check the median-continuum log-flux path in the new H5.

Run:  python verify_med_h5.py [path/to/all_spectra_float32_v2.h5]

Checks, in order:
  1. new root train-stat attrs exist  (norm_mean_log_med, norm_std_log_med)
  2. log_scale_flux_med exists in every split AND is length-aligned with
     obj_id / log_scale_flux  (this is the resize-omission bug from before)
  3. log_scale_flux_med is not accidentally identical to log_scale_flux
     (i.e. NORM_CMD actually differed from NORM_CMN in the derivation)
  4. NORM_CMD scalars are all finite & > 0  (skip-condition should have
     dropped anything else — non-zero here means the skip didn't bite)
  5. stored train stats (f8-accumulated) match a recompute from the stored
     f4 data, for BOTH mean- and median-normalised flux

Exit code 0 = all passed, 1 = something failed.
"""
import sys
import numpy as np
import h5py

DEFAULT_H5 = "all_spectra_float32_v2.h5"
SPLITS = ["train", "validation", "test"]


def approx(a, b, rtol=1e-2):
    """f8-attr vs f4-data recompute: generous tol, rounding-only differences expected."""
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-30)


def rel(a, b):
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def main(path=DEFAULT_H5):
    ok = True
    with h5py.File(path, "r") as hf:

        # --- 1. root attrs -------------------------------------------------
        print("== root attributes ==")
        for k in ["norm_mean_log", "norm_std_log",
                  "norm_mean_log_med", "norm_std_log_med"]:
            present = k in hf.attrs
            ok &= present
            print(f"  {k:20s}: {hf.attrs[k] if present else 'MISSING'}")

        # --- per-split checks ---------------------------------------------
        for split in SPLITS:
            print(f"\n== {split} ==")
            if split not in hf:
                print("  MISSING group"); ok = False; continue
            g = hf[split]

            # 2. existence + alignment
            missing = [n for n in ("log_scale_flux", "log_scale_flux_med", "obj_id")
                       if n not in g]
            if missing:
                print(f"  MISSING datasets: {missing}"); ok = False
                if "log_scale_flux_med" not in g:
                    continue

            n_id = g["obj_id"].shape[0]
            n_mn = g["log_scale_flux"].shape[0]
            n_md = g["log_scale_flux_med"].shape[0]
            aligned = (n_id == n_mn == n_md)
            ok &= aligned
            print(f"  rows: obj_id={n_id}  log_mean={n_mn}  log_med={n_md}  "
                  f"-> {'aligned' if aligned else 'MISALIGNED (resize list bug?)'}")
            if not aligned:
                continue

            # 3. med must differ from mean (sample first ~50 rows, cheap)
            s = slice(0, min(n_md, 50))
            a = g["log_scale_flux"][s]
            b = g["log_scale_flux_med"][s]
            both = ~np.isnan(a) & ~np.isnan(b)
            if both.sum() == 0:
                print("  !! no overlapping finite pixels to compare (tiny/empty split?)")
            elif np.array_equal(a[both], b[both]):
                print("  !! log_scale_flux_med IDENTICAL to log_scale_flux "
                      "(NORM_CMD == NORM_CMN?)"); ok = False
            else:
                print(f"  med differs from mean  (median |Δ| = "
                      f"{np.nanmedian(np.abs(a[both] - b[both])):.4g})  OK")

            # 4. NORM_CMD sanity
            if "NORM_CMD" in g:
                cmd = g["NORM_CMD"][:]
                bad = int(np.sum(~np.isfinite(cmd) | (cmd <= 0)))
                ok &= (bad == 0)
                print(f"  NORM_CMD: {bad} non-finite/<=0  "
                      f"{'OK' if bad == 0 else 'UNEXPECTED (skip cond should drop these)'}")

        # --- 5. recompute train stats from stored data --------------------
        # Loads the full train flux arrays into RAM (fine given your preload modes).
        print("\n== train stat recompute  (stored f8 attr  vs  f4-data recompute) ==")
        g = hf["train"]
        for dset_name, mean_attr, std_attr in [
            ("log_scale_flux",     "norm_mean_log",     "norm_std_log"),
            ("log_scale_flux_med", "norm_mean_log_med", "norm_std_log_med"),
        ]:
            if dset_name not in g or mean_attr not in hf.attrs:
                print(f"  skip {dset_name} (dataset or attr missing)"); ok = False; continue
            flat = g[dset_name][:]
            flat = flat[~np.isnan(flat)]          # gaps stored as NaN
            rc_mean, rc_std = float(np.mean(flat)), float(np.std(flat))  # ddof=0 = population
            s_mean, s_std = float(hf.attrs[mean_attr]), float(hf.attrs[std_attr])
            m_ok, s_ok = approx(rc_mean, s_mean), approx(rc_std, s_std)
            ok &= m_ok & s_ok
            print(f"  {dset_name}")
            print(f"    mean  stored={s_mean:+.6g}  recompute={rc_mean:+.6g}  "
                  f"relΔ={rel(rc_mean, s_mean):.2e}  {'OK' if m_ok else 'MISMATCH'}")
            print(f"    std   stored={s_std:+.6g}  recompute={rc_std:+.6g}  "
                  f"relΔ={rel(rc_std, s_std):.2e}  {'OK' if s_ok else 'MISMATCH'}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_H5))