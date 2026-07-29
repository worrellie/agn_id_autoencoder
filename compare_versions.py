"""
compare_versions.py — diff two H5 versions matched by obj_id (NOT row index,
because skip sets can differ between versions and shift the rows).

Two intended uses:

  # 1. v1 vs v2 : did adding the median path change the MEAN-normalised flux?
  #    Expect: log_scale_flux IDENTICAL on shared IDs.
  #    A few only-in-v1 IDs are OK (v2's tighter skip drops NORM_CMD<=0 spectra).
  compare("all_spectra_float32.h5", "all_spectra_float32_v2.h5",
          flux_key="log_scale_flux",
          same_scalars=[("NORM_CON", "NORM_CMN")])   # note the RENAME across versions

  # 2. v2 vs v3 : is v3 really "v2 + new SNR", or did the norm factor move too?
  #    Expect: NORM_CMN / log_scale_flux IDENTICAL, SNR_* DIFFERENT.
  #    If NORM_CMN DIFFERS -> the SNR edit leaked into the normalisation (confound).
  compare("all_spectra_float32_v2.h5", "all_spectra_float32_v3.h5",
          flux_key="log_scale_flux_med",
          same_scalars=["NORM_CMN", "NORM_CMD"],
          differ_scalars=["SNR_MEAN", "SNR_MED"])
"""
import sys
import numpy as np
import h5py

SPLITS = ("train", "validation", "test")


def _ids(g):
    return np.array([s.decode() if isinstance(s, bytes) else s for s in g["obj_id"][:]])


def _pair(k):
    return k if isinstance(k, tuple) else (k, k)


def _rows_for(ids, wanted):
    idx = {i: r for r, i in enumerate(ids)}
    return [idx[i] for i in wanted]


def compare(path_a, path_b, flux_key,
            same_scalars=(), differ_scalars=(), atol=0.0):
    ok = True
    with h5py.File(path_a, "r") as ha, h5py.File(path_b, "r") as hb:
        print(f"A = {path_a}\nB = {path_b}\n")

        for split in SPLITS:
            if split not in ha or split not in hb:
                print(f"== {split} ==  MISSING in one file"); ok = False; continue
            ga, gb = ha[split], hb[split]

            ia, ib = _ids(ga), _ids(gb)
            sa, sb = set(ia), set(ib)
            shared = sorted(sa & sb)
            print(f"== {split} ==")
            print(f"  rows: A={len(ia)} B={len(ib)}  shared={len(shared)}  "
                  f"only_A={len(sa - sb)}  only_B={len(sb - sa)}")
            if not shared:
                print("  no shared IDs — cannot compare"); ok = False; continue

            ra, rb = _rows_for(ia, shared), _rows_for(ib, shared)

            # ---- flux on shared IDs (load full, then fancy-index; fits your RAM) ----
            if flux_key in ga and flux_key in gb:
                fa = ga[flux_key][:][ra]
                fb = gb[flux_key][:][rb]
                # equal_nan: gaps are NaN in the same positions if unchanged
                eq = np.isclose(fa, fb, rtol=0, atol=atol, equal_nan=True)
                n_diff = int((~eq).sum())
                if n_diff == 0:
                    print(f"  {flux_key}: IDENTICAL on shared IDs")
                else:
                    ok = False
                    with np.errstate(invalid="ignore"):
                        mad = float(np.nanmax(np.abs(fa - fb)))
                    # NaN-position mismatch = gap structure changed
                    nan_mismatch = int((np.isnan(fa) ^ np.isnan(fb)).sum())
                    print(f"  {flux_key}: DIFFERS — {n_diff} pixels (max|Δ|={mad:.3e}, "
                          f"nan-position mismatches={nan_mismatch})")
            else:
                print(f"  {flux_key}: missing in one file"); ok = False

            # ---- scalars expected to MATCH ----
            for key in same_scalars:
                ka, kb = _pair(key)
                if ka in ga and kb in gb:
                    va, vb = ga[ka][:][ra], gb[kb][:][rb]
                    same = np.allclose(va, vb, rtol=0, atol=atol, equal_nan=True)
                    mad = float(np.nanmax(np.abs(va - vb)))
                    print(f"  {ka}->{kb}: {'IDENTICAL' if same else f'DIFFERS (max|Δ|={mad:.3e})  <-- unexpected'}")
                    ok &= same
                else:
                    print(f"  {ka}->{kb}: missing in one file")

            # ---- scalars expected to DIFFER (the intended change) ----
            for key in differ_scalars:
                ka, kb = _pair(key)
                if ka in ga and kb in gb:
                    va, vb = ga[ka][:][ra], gb[kb][:][rb]
                    same = np.allclose(va, vb, rtol=0, atol=atol, equal_nan=True)
                    mad = float(np.nanmax(np.abs(va - vb)))
                    if same:
                        print(f"  {ka}->{kb}: UNCHANGED  <-- expected this to differ")
                        ok = False
                    else:
                        print(f"  {ka}->{kb}: differs as expected (max|Δ|={mad:.3e})")
                else:
                    print(f"  {ka}->{kb}: missing in one file")
            print()

        # ---- stored train stats ----
        print("== train stats (root attrs) ==")
        for k in ("norm_mean_log", "norm_std_log",
                  "norm_mean_log_med", "norm_std_log_med"):
            va, vb = ha.attrs.get(k), hb.attrs.get(k)
            if va is None or vb is None:
                print(f"  {k}: A={va} B={vb}  (absent in one)")
            else:
                same = np.isclose(float(va), float(vb))
                print(f"  {k}: A={float(va):.6g} B={float(vb):.6g}  "
                      f"{'same' if same else 'DIFFER (expected if skip set changed)'}")

    print("\n" + ("CONSISTENT with expectations" if ok else "CHECK THE FLAGGED LINES"))
    return 0 if ok else 1


if __name__ == "__main__":

    print("\n\n### v2 vs v3 : SNR-only, or flux confounded? ###\n")
    compare("all_spectra_float32_v2.h5", "all_spectra_float32_v3.h5",
            flux_key="log_scale_flux_med",
            same_scalars=["NORM_CMN", "NORM_CMD"],
            differ_scalars=["SNR_MEAN", "SNR_MED"])
    
    # edit these two calls to point at your actual filenames
    print("### v1 vs v2 : mean flux unchanged? ###\n")
    compare("all_spectra_float32.h5", "all_spectra_float32_v2.h5",
            flux_key="log_scale_flux",
            same_scalars=[("NORM_CON", "NORM_CMN")])

