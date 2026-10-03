#!/usr/bin/env python3
"""
Plot every data extension of one MOONS simulated FITS file in a near-square grid.

Each 1D extension is plotted against the vacuum wavelengths (HDU 9); the
wavelength extension itself is plotted against pixel index.

Usage:
    python plot_fits_extensions.py cosmos_bagpipes_202599_2h_z0_9110_RI.fits [--out ri_extensions.png]
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

WAVE_HDU = 9
CLIP = {"Flux-calib frame"}   # sky residuals squash these; clip y to 0.5–99.5 percentile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fits_file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with fits.open(args.fits_file) as h:
        band = h[0].header.get("BAND", "").strip()
        wave = h[WAVE_HDU].data
        exts = [(i, h[i].header.get("NAME", f"HDU {i}").strip(),
                 h[i].header.get("TUNIT2", "").strip(), h[i].data)
                for i in range(1, len(h)) if h[i].data is not None]

    n = len(exts)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.4 * nrows), squeeze=False)

    for ax, (i, name, unit, y) in zip(axes.flat, exts):
        if i == WAVE_HDU:
            ax.plot(np.arange(y.size), y, color="tab:blue", lw=1)
            ax.set_xlabel("pixel index")
        else:
            if name == "Sky mask":
                ax.step(wave, y, where="mid", color="tab:blue", lw=0.6)
                ax.set_title(f"HDU {i}: {name}  ({y.mean():.1%} flagged)", fontsize=10)
            else:
                ax.plot(wave, y, color="tab:blue", lw=0.4)
            ax.set_xlabel("Vacuum wavelength [Å]")
            ax.set_xlim(wave[0], wave[-1])
        if name in CLIP:
            lo, hi = np.nanpercentile(y, [0.5, 99.5])
            pad = 0.1 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            name += " (y clipped)"
        if not ax.get_title():
            ax.set_title(f"HDU {i}: {name}", fontsize=10)
        ax.set_ylabel(unit or "value", fontsize=8)
        ax.tick_params(labelsize=8)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    stem = os.path.basename(args.fits_file)
    fig.suptitle(f"{stem}  ({band} band, all extensions)", fontsize=12)
    fig.tight_layout()
    out = args.out or f"{band or 'fits'}_extensions.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
