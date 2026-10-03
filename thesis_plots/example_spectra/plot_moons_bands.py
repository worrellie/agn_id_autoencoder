#!/usr/bin/env python3
"""
Plot flux vs wavelength for the three MOONS bands (RI, YJ, H) of one simulated source.

Usage:
    python plot_moons_bands.py cosmos_bagpipes_202599_2h_z0_9110 [--out spectrum.png]
    python plot_moons_bands.py cosmos_bagpipes_202599_2h_z0_9110 --combined   # all bands on one axis
    (the argument is the file stem; _RI.fits, _YJ.fits and _H.fits are appended)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

BANDS = ["RI", "YJ", "H"]
FLUX_HDU, TMPL_HDU, WAVE_HDU = 1, 4, 9   # flux-calib frame, noiseless template, vacuum wavelengths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem")
    ap.add_argument("--out", default=None)
    ap.add_argument("--combined", action="store_true",
                    help="plot all three bands on one axis, shading the gaps between them")
    args = ap.parse_args()
    if args.combined:
        return plot_combined(args.stem, args.out or "spectrum_combined.png")
    args.out = args.out or "spectrum_bands.png"

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    for ax, band in zip(axes, BANDS):
        with fits.open(f"{args.stem}_{band}.fits") as h:
            wave = h[WAVE_HDU].data
            flux = h[FLUX_HDU].data
            tmpl = h[TMPL_HDU].data
            R = h[0].header["R"]
        # template drawn first (lower zorder) so the noisy flux sits on top of it
        ax.plot(wave, tmpl, color="tab:red", lw=1, zorder=1, label="noiseless template")
        ax.plot(wave, flux, color="k", lw=0.5, alpha=0.5, zorder=2, label="flux (with noise)")
        # clip y-limits to the bulk of the data so sky residuals don't squash the plot
        lo, hi = np.nanpercentile(flux, [0.5, 99.5])
        pad = 0.1 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(wave[0], wave[-1])
        ax.set_title(f"{band}  ({wave[0]:.0f}–{wave[-1]:.0f} Å, R = {R:.0f})", fontsize=10)
        ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=8)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Observed wavelength [Å]")
    fig.suptitle(args.stem.split("/")[-1], fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


BAND_COLOURS = {"RI": "tab:blue", "YJ": "tab:green", "H": "tab:purple"}


def plot_combined(stem, out):
    data = {}
    for band in BANDS:
        with fits.open(f"{stem}_{band}.fits") as h:
            data[band] = (h[WAVE_HDU].data, h[FLUX_HDU].data, h[TMPL_HDU].data)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    for band, (wave, flux, tmpl) in data.items():
        ax.plot(wave, tmpl, color="tab:red", lw=1, zorder=1,
                label="noiseless template" if band == BANDS[0] else None)
        ax.plot(wave, flux, color="k", lw=0.4, alpha=0.5, zorder=2,
                label="flux (with noise)" if band == BANDS[0] else None)
        # thin coloured bar along the top marking each band's coverage
        ax.axvspan(wave[0], wave[-1], ymin=0.97, ymax=1.0, color=BAND_COLOURS[band], lw=0, zorder=3)
        ax.text(0.5 * (wave[0] + wave[-1]), 0.935, band, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=BAND_COLOURS[band], weight="bold")

    # shade any gaps in coverage between consecutive bands
    for b1, b2 in zip(BANDS[:-1], BANDS[1:]):
        g0, g1 = data[b1][0][-1], data[b2][0][0]
        if g1 - g0 > 5:          # ignore sub-pixel joins (RI/YJ meet at ~9340 Å)
            ax.axvspan(g0, g1, color="0.85", zorder=0, lw=0)
            ax.text(0.5 * (g0 + g1), 0.5, f"no coverage\n{g0:.0f}–{g1:.0f} Å",
                    transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=8, color="0.3", rotation=90)

    allflux = np.concatenate([d[1] for d in data.values()])
    lo, hi = np.nanpercentile(allflux, [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.5 * pad)          # extra headroom for band labels
    ax.set_xlim(data[BANDS[0]][0][0], data[BANDS[-1]][0][-1])
    ax.set_xlabel("Observed wavelength [Å]")
    ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax.set_title(stem.split("/")[-1], fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
