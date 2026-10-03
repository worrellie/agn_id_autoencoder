#!/usr/bin/env python3
"""
Top: the original MOONS simulated spectrum, all three bands (RI, YJ, H) joined, unedited.
Bottom: the final processed spectrum (the *_noisy_deZ_rebinned.fits table), which the
pipeline shifts from the source redshift to a common reference redshift (z_ref = 0.9)
and rebins onto a 4 Å grid.
Third panel: the same processed spectrum in the rest frame:
    λ_rest = λ_zref / (1 + z_ref),   F_λ,rest = F_λ,zref × (1 + z_ref)
(the flux factor keeps F_λ dλ, i.e. integrated line fluxes, unchanged by the stretch).

Usage:
    python plot_original_vs_processed.py cosmos_bagpipes_202599_2h_z0_9110 [--out orig_vs_proc.png]
    (stem: the script appends _RI/_YJ/_H.fits and _noisy_deZ_rebinned.fits)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

BANDS = ["RI", "YJ", "H"]
FLUX_HDU, WAVE_HDU = 1, 9          # 'Flux-calib frame', 'Vacuum wavelengths'

# rest-frame vacuum wavelengths (Å) of lines marked on the rest-frame panel
LINES = {"[OII]": 3728.5, "Hδ": 4102.9, "Hγ": 4341.7, "Hβ": 4862.7, "[OIII]": 5008.2,
         "[OI]": 6302.0, "Hα": 6564.6, "[SII]": 6725.0}


def ylims(ax, y, lo_pct=0.5, hi_pct=99.5):
    lo, hi = np.nanpercentile(y, [lo_pct, hi_pct])
    pad = 0.1 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem")
    ap.add_argument("--out", default="original_vs_processed.png")
    ap.add_argument("--zref", type=float, default=0.9,
                    help="reference redshift the processed spectra are shifted to (default 0.9)")
    args = ap.parse_args()

    # original bands
    bands = []
    for b in BANDS:
        with fits.open(f"{args.stem}_{b}.fits") as h:
            bands.append((b, h[WAVE_HDU].data, h[FLUX_HDU].data))

    # processed spectrum
    with fits.open(f"{args.stem}_noisy_deZ_rebinned.fits") as h:
        tab, hdr = h[1].data, h[1].header
        lam_p, flux_p = tab["lambda"], tab["flux"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10.5))

    # --- top: original, all bands, as delivered
    for b, w, f in bands:
        ax1.plot(w, f, color="tab:blue", lw=0.4)
        ax1.text(0.5 * (w[0] + w[-1]), 0.95, b, transform=ax1.get_xaxis_transform(),
                 ha="center", va="top", fontsize=9, weight="bold")
    for (_, w1, _), (_, w2, _) in zip(bands[:-1], bands[1:]):
        if w2[0] - w1[-1] > 5:
            ax1.axvspan(w1[-1], w2[0], color="0.85", lw=0, zorder=0)
            ax1.text(0.5 * (w1[-1] + w2[0]), 0.5, f"no coverage\n{w1[-1]:.0f}–{w2[0]:.0f} Å",
                     transform=ax1.get_xaxis_transform(), ha="center", va="center",
                     fontsize=8, color="0.3", rotation=90)
    ylims(ax1, np.concatenate([f for _, _, f in bands]))
    ax1.set_xlim(bands[0][1][0], bands[-1][1][-1])
    ax1.set_xlabel("Observed vacuum wavelength [Å]")
    ax1.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    npix = sum(w.size for _, w, _ in bands)
    ax1.set_title(f"Original MOONS simulation at z = {hdr.get('REDSHIF', float('nan')):.3f} "
                  f"(RI + YJ + H, {npix} pixels, native sampling)",
                  fontsize=10)

    # --- bottom: processed
    ax2.plot(lam_p, flux_p, color="tab:orange", lw=0.6)
    ylims(ax2, flux_p)
    ax2.set_xlim(lam_p[0], lam_p[-1])
    ax2.set_xlabel(f"Wavelength at reference redshift z$_{{ref}}$ = {args.zref:g} [Å]")
    ax2.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    dl = np.median(np.diff(lam_p))
    z = hdr.get("REDSHIF", float("nan"))
    ax2.set_title(f"Processed: shifted from z = {z:.3f} to z$_{{ref}}$ = {args.zref:g}, "
                  f"rebinned to {dl:.0f} Å ({lam_p.size} pixels)", fontsize=10)

    # --- third: processed spectrum in the rest frame
    lam_rest = lam_p / (1 + args.zref)
    flux_rest = flux_p * (1 + args.zref)      # F_λ per rest-frame Å
    ax3.plot(lam_rest, flux_rest, color="tab:green", lw=0.6)
    ylims(ax3, flux_rest)
    ax3.set_xlim(lam_rest[0], lam_rest[-1])
    for name, lr in LINES.items():
        if lam_rest[0] < lr < lam_rest[-1]:
            ax3.axvline(lr, color="0.5", ls="--", lw=0.7, zorder=0)
            ax3.text(lr, 0.97, name, transform=ax3.get_xaxis_transform(), rotation=90,
                     ha="right", va="top", fontsize=8, color="0.3")
    ax3.set_xlabel("Rest-frame wavelength [Å]")
    ax3.set_ylabel(r"$F_{\lambda,\mathrm{rest}}$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax3.set_title(f"Processed, rest frame (λ / (1 + {args.zref:g}), F$_\\lambda$ × (1 + {args.zref:g}); "
                  f"{np.median(np.diff(lam_rest)):.2f} Å bins)", fontsize=10)

    fig.suptitle(args.stem.split("/")[-1], fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
