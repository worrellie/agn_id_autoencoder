"""
Reconstruction error vs REDSHIFT — with the SNR confound handled.

Run from the directory containing the RUN_* folders:
    uv run python plot_loss_vs_z.py
    uv run python plot_loss_vs_z.py RUN_a RUN_b RUN_c

Produces loss_vs_z.png:
  TOP    raw reconstruction error vs z
  BOTTOM noise-normalised chi2 vs z

WHY TWO ROWS.
  We already know loss ~ noise ~ 1/SNR (rho = -0.97). So ANY apparent trend of loss
  with redshift may be nothing to do with redshift: high-z objects are typically
  fainter, so they have lower SNR, so they have higher loss. That is a selection
  effect, not physics.

  The script prints rho(z, SNR) FIRST. That is the confound. If it is strongly
  negative, the top row tells you almost nothing on its own and you must read the
  bottom row, where the noise dependence has been divided out.

  rho(z, chi2) ~ 0        -> no genuine redshift dependence once noise is accounted for
  rho(z, chi2) far from 0 -> a REAL trend: the model reconstructs some redshifts
                             worse than others, beyond what noise explains.

THINGS THAT COULD CAUSE A REAL z-TREND (worth knowing before interpreting):
  1. Rest-frame coverage. De-redshifting shifts which observed wavelengths land on
     the common grid, so different z sample different rest-frame features.
  2. Gap positions. The RI/YJ/H inter-band gaps move with z, so the number and
     location of masked pixels is z-dependent. Check n_unmasked vs z if a trend shows.
  3. Sample imbalance. If the training set is sparse at some redshifts, the model
     will have learned those less well. That is a real, reportable finding.
"""

import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SNR_FLOOR = 1e-3
N_BINS = 12


def short(name):
    bits = name.split("_")
    nl = next((b for b in bits if b.startswith("nl")), "?")
    ls = next((b for b in bits if b.startswith("ls")), "?")
    act = next((b for b in bits if b in ("ReLU", "Tanh", "LeakyReLU")), "?")
    return f"{nl} {ls} {act}"


def load(run_dir):
    hits = glob.glob(f"{run_dir}/*_validation_latent.npz")
    if not hits:
        raise FileNotFoundError(f"no *_validation_latent.npz in {run_dir}")
    d = np.load(hits[0])
    if "redshift" not in d:
        raise KeyError(f"{hits[0]} has no 'redshift' array — keys: {list(d.keys())}")
    return (np.asarray(d["redshift"], float),
            np.asarray(d["snr"], float),
            np.asarray(d["loss_unscaled"], float))


def binned_median(x, y, nbins=N_BINS):
    """Median of y in equal-count bins of x. Robust to the long loss tail."""
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    centres, meds, los, his = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 5:
            continue
        centres.append(np.median(x[m]))
        meds.append(np.median(y[m]))
        los.append(np.percentile(y[m], 16))
        his.append(np.percentile(y[m], 84))
    return map(np.array, (centres, meds, los, his))


def main(run_dirs):
    n = len(run_dirs)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9), squeeze=False)

    # ---- the confound, computed once (same data for every model) -------------
    z0, s0, _ = load(run_dirs[0])
    k0 = (s0 > SNR_FLOOR) & np.isfinite(z0)
    rho_z_snr = spearmanr(z0[k0], s0[k0])[0]
    print(f"\nCONFOUND CHECK:  rho(z, SNR) = {rho_z_snr:+.3f}")
    if abs(rho_z_snr) > 0.3:
        print("  -> z and SNR are correlated. The TOP row is contaminated by this.")
        print("     Read the BOTTOM row (chi2) for any genuine redshift dependence.")
    else:
        print("  -> z and SNR are largely independent. The top row is interpretable.")
    print()

    print(f"{'model':<22} {'rho(z,loss)':>12} {'rho(z,chi2)':>12}")
    print("-" * 48)

    for j, rd in enumerate(run_dirs):
        z, snr, loss = load(rd)
        keep = (snr > SNR_FLOOR) & np.isfinite(loss) & (loss > 0) & np.isfinite(z)
        z, snr, loss = z[keep], snr[keep], loss[keep]

        chi2 = loss * snr**2          # noise-normalised: "bigger than the noise allows?"

        rho_raw = spearmanr(z, loss)[0]
        rho_chi = spearmanr(z, chi2)[0]
        print(f"{short(rd):<22} {rho_raw:12.3f} {rho_chi:12.3f}")

        # ---------------- TOP: raw loss vs z ----------------
        ax = axes[0][j]
        ax.hexbin(z, loss, yscale="log", gridsize=55, cmap="viridis",
                  mincnt=1, linewidths=0)
        c, m_, lo, hi = binned_median(z, loss)
        ax.plot(c, m_, "r-", lw=2.5, label="binned median")
        ax.fill_between(c, lo, hi, color="r", alpha=0.20, label="16–84%")
        ax.set_xlabel("redshift $z$")
        ax.set_ylabel("reconstruction MSE (physical)")
        ax.set_title(f"{short(rd)}\n" + r"raw:  $\rho$ = " + f"{rho_raw:.3f}", fontsize=10)
        ax.legend(fontsize=8)

        # ---------------- BOTTOM: chi2 vs z ----------------
        ax = axes[1][j]
        ax.hexbin(z, chi2, yscale="log", gridsize=55, cmap="magma",
                  mincnt=1, linewidths=0)
        c, m_, lo, hi = binned_median(z, chi2)
        ax.plot(c, m_, "c-", lw=2.5, label="binned median")
        ax.fill_between(c, lo, hi, color="c", alpha=0.20, label="16–84%")
        ax.set_xlabel("redshift $z$")
        ax.set_ylabel(r"$\chi^2 \approx$ MSE $\times$ SNR$^2$")
        ax.set_title(r"noise-normalised:  $\rho$ = " + f"{rho_chi:.3f}", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"Reconstruction error vs redshift    "
                 f"[confound: " + r"$\rho$(z, SNR) = " + f"{rho_z_snr:+.3f}]",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig("loss_vs_z.png", dpi=150)
    print("\nwrote loss_vs_z.png")

    print("""
HOW TO READ IT
  Compare the two rows. If the top row shows a trend and the bottom row is FLAT,
  the apparent redshift dependence was just the SNR confound — high-z objects are
  fainter, not harder.

  If the BOTTOM row still trends, that is a real result: the model reconstructs some
  redshifts worse than others, beyond what noise explains. Then check whether it
  tracks rest-frame coverage / gap positions / training-set density in z.
""")


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(glob.glob("RUN_*"))
    if not runs:
        sys.exit("no RUN_* directories found — run from the folder containing them")
    main(runs)