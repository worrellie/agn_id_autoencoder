"""
Reconstruction error vs SNR — the diagnostic behind rho(SNR, loss) = -0.97.

Run from the directory containing the RUN_* folders:
    uv run python plot_loss_vs_snr.py
    uv run python plot_loss_vs_snr.py RUN_a RUN_b RUN_c     # or name them explicitly

Produces loss_vs_snr.png with two rows:
  TOP    raw reconstruction error vs SNR   -> shows the problem
  BOTTOM noise-normalised chi2 vs SNR      -> shows whether normalising fixes it

THE KEY NUMBER is the fitted slope in the top row.

  Continuum-normalised flux sits at ~1, so the per-pixel noise is sigma ~ 1/SNR.
  If the model reconstructs the SIGNAL perfectly and leaves only NOISE behind, then
        MSE = <residual^2> = sigma^2 ~ SNR^-2
  i.e. on log-log axes, a straight line of SLOPE -2.

  slope ~ -2  =>  the loss IS the noise variance. The autoencoder is measuring SNR
                  and nothing else. No amount of extra capacity will change this.
  slope  > -2  =>  something beyond noise contributes (model error, real outliers).
                  That excess is where any genuine anomaly signal lives.
"""

import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SNR_FLOOR = 1e-3   # drop non-positive / absurd SNR (log axes, and 1/SNR blows up)


def short(name):
    """RUN_StandardAutoencoder_nl1_ls256_..._ReLU_...  ->  nl1 ls256 ReLU"""
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
    return np.asarray(d["snr"], float), np.asarray(d["loss_unscaled"], float)


def main(run_dirs):
    n = len(run_dirs)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9), squeeze=False)

    print(f"{'model':<22} {'slope':>7} {'rho(raw)':>9} {'rho(chi2)':>10} "
          f"{'chi2 med':>9} {'chi2 p99':>9}")
    print("-" * 72)

    for j, rd in enumerate(run_dirs):
        snr, loss = load(rd)

        keep = (snr > SNR_FLOOR) & np.isfinite(loss) & (loss > 0)
        s, l = snr[keep], loss[keep]
        n_drop = len(snr) - keep.sum()

        # --- fit log(loss) = m * log(snr) + c ; m = -2 means "pure noise" -------
        m, c = np.polyfit(np.log10(s), np.log10(l), 1)

        # --- noise-normalised score: chi2 ~ MSE / sigma^2 = MSE * SNR^2 --------
        chi2 = l * s**2

        rho_raw = spearmanr(s, l)[0]
        rho_chi = spearmanr(s, chi2)[0]

        print(f"{short(rd):<22} {m:7.2f} {rho_raw:9.3f} {rho_chi:10.3f} "
              f"{np.median(chi2):9.3g} {np.percentile(chi2, 99):9.3g}")

        # ================= TOP: raw loss vs SNR =================
        ax = axes[0][j]
        ax.hexbin(s, l, xscale="log", yscale="log", gridsize=60,
                  cmap="viridis", mincnt=1, linewidths=0)

        xs = np.logspace(np.log10(s.min()), np.log10(s.max()), 50)
        ax.plot(xs, 10**c * xs**m, "r-", lw=2,
                label=f"fit: slope {m:.2f}")
        # anchor the reference line at the data median so it's comparable
        anchor = np.median(l) / np.median(s)**-2
        ax.plot(xs, anchor * xs**-2.0, "w--", lw=2,
                label="pure noise: slope $-2$")

        ax.set_xlabel("SNR")
        ax.set_ylabel("reconstruction MSE (physical)")
        ax.set_title(f"{short(rd)}\n" + r"$\rho$ = " + f"{rho_raw:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

        # ================= BOTTOM: chi2 vs SNR =================
        ax = axes[1][j]
        ax.hexbin(s, chi2, xscale="log", yscale="log", gridsize=60,
                  cmap="magma", mincnt=1, linewidths=0)
        ax.axhline(np.median(chi2), color="cyan", ls="--", lw=2,
                   label=f"median {np.median(chi2):.3g}")
        ax.set_xlabel("SNR")
        ax.set_ylabel(r"$\chi^2 \approx$ MSE $\times$ SNR$^2$")
        ax.set_title(r"noise-normalised   $\rho$ = " + f"{rho_chi:.3f}", fontsize=10)
        ax.legend(fontsize=8)

        if j == 0 and n_drop:
            print(f"   (dropped {n_drop} spectra with SNR <= {SNR_FLOOR})")

    fig.suptitle("Reconstruction error vs SNR — is the model measuring signal, or noise?",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig("loss_vs_snr.png", dpi=150)
    print("\nwrote loss_vs_snr.png")

    print("""
HOW TO READ IT
  Top row, fitted slope:
     ~ -2  the loss is pure noise variance. The AE reconstructs the signal fine and
           what is left over is the noise. Ranking by raw MSE ranks by SNR, nothing more.
     > -2  there is excess error beyond noise. THAT excess is the anomaly signal.

  Bottom row, rho(chi2):
     ~ 0   normalising by the noise removed the SNR dependence. Use chi2 as the
           anomaly score, and whatever sits at high chi2 is a genuine outlier.
     still strongly negative -> a per-spectrum scalar SNR is too crude. You would
           need per-pixel errors (an error / ivar column from the FITS files).
""")


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(glob.glob("RUN_*"))
    if not runs:
        sys.exit("no RUN_* directories found — run this from the folder containing them")
    main(runs)