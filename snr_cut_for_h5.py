#!/usr/bin/env python3
"""
Cut HDF5 spectra files to sources with SNR >= a threshold (default SNR_MEAN >= 1).

For every split group (training / validation / test ...):
  - per-source datasets (first dimension = number of sources, e.g. raw_flux, obj_id,
    redshift, SNR_*) are filtered with the same mask, so rows stay aligned;
  - other datasets (e.g. skipped_norm_con, which lists spectra already excluded) are
    copied unchanged.
File and group attributes (e.g. the root 'wavelengths') are copied, and compression is
kept. Sources with a NaN SNR are dropped (NaN >= 1 is False).

The originals are never modified: output goes next to each input as
<name>_<key>_ge<threshold>.h5 (e.g. all_agn_float32_SNR_MEAN_ge1.h5).

Usage:
    python snr_cut.py all_agn_float32.h5 galaxies.h5
    python snr_cut.py a.h5 b.h5 --snr-key SNR_MED --min-snr 3
"""
import argparse
import os

import h5py
import numpy as np

NOT_PER_SOURCE = {"skipped_norm_con"}   # never filtered, even if its length happens to match


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def write_dataset(g_out, name, d, data):
    kw = {}
    if d.compression and data.shape[0] > 0:
        kw["compression"] = d.compression
        kw["compression_opts"] = d.compression_opts
        # keep one-spectrum-per-chunk for 2D flux arrays; let h5py choose for 1D
        kw["chunks"] = (1,) + data.shape[1:] if data.ndim == 2 else True
    out = g_out.create_dataset(name, data=data, **kw)
    copy_attrs(d, out)


def cut_file(path, key, thr, overwrite):
    stem, ext = os.path.splitext(path)
    out_path = f"{stem}_{key}_ge{thr:g}{ext}"
    if os.path.exists(out_path) and not overwrite:
        raise SystemExit(f"{out_path} exists; use --overwrite to replace it")

    print(f"\n{os.path.basename(path)} -> {os.path.basename(out_path)}")
    with h5py.File(path, "r") as fin, h5py.File(out_path, "w") as fout:
        copy_attrs(fin, fout)
        fout.attrs["snr_cut"] = f"{key} >= {thr:g}"
        fout.attrs["snr_cut_source_file"] = os.path.basename(path)

        for name, item in fin.items():
            if isinstance(item, h5py.Dataset):          # root-level dataset: copy as is
                fin.copy(item, fout, name=name)
                continue
            if key not in item:
                print(f"  {name}: no '{key}' dataset, copied unchanged")
                fin.copy(item, fout, name=name)
                continue

            snr = item[key][()]
            n = snr.size
            mask = snr >= thr                           # NaN -> False
            g_out = fout.create_group(name)
            copy_attrs(item, g_out)
            for dname, d in item.items():
                if not isinstance(d, h5py.Dataset):
                    item.copy(d, g_out, name=dname)
                elif dname not in NOT_PER_SOURCE and d.ndim >= 1 and d.shape[0] == n:
                    write_dataset(g_out, dname, d, d[()][mask])
                else:
                    write_dataset(g_out, dname, d, d[()])
            n_nan = int(np.isnan(snr).sum()) if snr.dtype.kind == "f" else 0
            print(f"  {name:<11} kept {mask.sum():>6d} / {n:<6d} ({100 * mask.mean():5.1f}%)"
                  + (f"  [{n_nan} NaN SNR dropped]" if n_nan else ""))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="HDF5 files to cut")
    ap.add_argument("--snr-key", default="SNR_MEAN")
    ap.add_argument("--min-snr", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = ap.parse_args()
    for p in args.files:
        cut_file(p, args.snr_key, args.min_snr, args.overwrite)


if __name__ == "__main__":
    main()
