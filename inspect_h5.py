"""Work out which transform + normaliser produced log_scale_flux and log_scale_flux_med.

Recomputes each stored dataset from raw_flux using every combination of
candidate transform and per-spectrum normaliser, and reports how closely each
matches. The combination with near-zero error is the forward transform, which
tells you exactly what the inversion in funcs.py needs to undo.

Usage:
    uv run python check_log_transform.py all_spectra_float32_v3.h5
"""
import sys

import h5py
import numpy as np

N_ROWS = 200  # a sample is enough to identify the transform

TRANSFORMS = {
    "log1p(x)": np.log1p,
    "sign(x)*log1p(|x|)": lambda x: np.sign(x) * np.log1p(np.abs(x)),
    "arcsinh(x)": np.arcsinh,
    "log10(x)": np.log10,
    "log(x)": np.log,
}
NORMS = ["NORM_CMN", "NORM_CMD", "NORM_MED"]
TARGETS = ["log_scale_flux", "log_scale_flux_med"]


def main(path):
    with h5py.File(path, "r") as hf:
        g = hf["train"]
        raw = g["raw_flux"][:N_ROWS].astype(np.float64)
        norms = {n: g[n][:N_ROWS].astype(np.float64)[:, None] for n in NORMS}

        for target in TARGETS:
            stored = g[target][:N_ROWS].astype(np.float64)
            print(f"\n{target}:")
            results = []
            with np.errstate(all="ignore"):
                for norm_name, norm in norms.items():
                    for t_name, fn in TRANSFORMS.items():
                        recomputed = fn(raw / norm)
                        ok = np.isfinite(stored) & np.isfinite(recomputed)
                        if ok.sum() == 0:
                            continue
                        coverage = ok.sum() / np.isfinite(stored).sum()
                        err = np.abs(recomputed[ok] - stored[ok]).max()
                        results.append((err, coverage, t_name, norm_name))
            for err, coverage, t_name, norm_name in sorted(results)[:5]:
                print(f"  {t_name:<22} raw_flux/{norm_name:<9} "
                      f"max abs diff={err:.3g}  coverage={coverage:.1%}")


if __name__ == "__main__":
    main(sys.argv[1])