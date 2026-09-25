"""List everything in an HDF5 file: groups, datasets (shape, dtype), attributes.

Usage:
    uv run python inspect_h5.py all_spectra_float32_v3.h5
    uv run python inspect_h5.py all_spectra_float32_v3.h5 --stats   # adds quick stats per dataset
"""
import argparse

import h5py
import numpy as np


def show_attrs(obj, indent):
    for key, val in obj.attrs.items():
        print(f"{indent}  @{key} = {val!r}")


def quick_stats(ds, n_rows=1000):
    """Stats on the first n_rows only, so large datasets aren't loaded into memory."""
    if ds.dtype.kind not in "fiu" or ds.size == 0:
        return ""
    sample = ds[:n_rows] if ds.ndim > 0 else ds[()]
    sample = np.asarray(sample, dtype=np.float64)
    nan_frac = np.isnan(sample).mean()
    return (f"  [first {min(n_rows, len(ds)) if ds.ndim else 1} rows: "
            f"min={np.nanmin(sample):.4g} max={np.nanmax(sample):.4g} "
            f"mean={np.nanmean(sample):.4g} NaN={nan_frac:.1%}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--stats", action="store_true", help="print quick stats per dataset")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as hf:
        print(f"{args.path}")
        show_attrs(hf, "")

        def visit(name, obj):
            depth = name.count("/")
            indent = "  " * (depth + 1)
            label = name.split("/")[-1]
            if isinstance(obj, h5py.Group):
                print(f"{indent}{label}/  (group, {len(obj)} items)")
            else:
                line = f"{indent}{label}  shape={obj.shape} dtype={obj.dtype}"
                if args.stats:
                    line += quick_stats(obj)
                print(line)
            show_attrs(obj, indent)

        hf.visititems(visit)


if __name__ == "__main__":
    main()