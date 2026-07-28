"""
Spearman agreement between log-scaled and physical-space reconstruction error.

Run from the directory containing the RUN_* folders:
    uv run python plot_metric_agreement.py
    uv run python plot_metric_agreement.py RUN_a RUN_b RUN_c

Produces metric_agreement.png with two rows:
  TOP    raw values:  scaled MSE vs unscaled MSE  (log-log)
  BOTTOM RANK vs RANK — this is literally what Spearman's rho measures

WHY THIS FIGURE EXISTS.
  The pipeline can score a spectrum two ways:
     loss_scaled   MSE in z-scored log space (the space the model trains in)
     loss_unscaled MSE in physical flux space (after inverting z-score and log1p)
  The anomaly ranking only ever uses the ORDER of these scores, never their values
  (np.argsort -> take top N). So the only question that matters is whether the two
  produce the same ordering. That is exactly what Spearman's rho measures, and the
  bottom row shows it directly: points on the diagonal = identical ranking.

  A high rho justifies choosing the numerically stable metric (log space) without
  loss of information — which is the claim this figure supports.
"""

import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

TOP_N = 100     # size of the "most anomalous" set whose overlap we quote


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
    return (np.asarray(d["loss_scaled"], float),
            np.asarray(d["loss_unscaled"], float))


def main(run_dirs):
    n = len(run_dirs)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 9.2), squeeze=False)

    print(f"{'model':<22} {'spearman':>9} {'pearson':>9} {'top-%d overlap' % TOP_N:>15}")
    print("-" * 60)

    for j, rd in enumerate(run_dirs):
        ls_, lu = load(rd)

        keep = np.isfinite(ls_) & np.isfinite(lu) & (ls_ > 0) & (lu > 0)
        ls_, lu = ls_[keep], lu[keep]

        rho = spearmanr(ls_, lu)[0]
        # Pearson on the RAW values for contrast: it is much lower, because the
        # relationship is monotonic but strongly non-linear (expm1). That gap is
        # precisely why Spearman is the right statistic here.
        r_lin = pearsonr(ls_, lu)[0]

        top_s = set(np.argsort(ls_)[-TOP_N:])
        top_u = set(np.argsort(lu)[-TOP_N:])
        overlap = len(top_s & top_u)

        print(f"{short(rd):<22} {rho:9.4f} {r_lin:9.4f} {overlap:12d}/{TOP_N}")

        # ================= TOP: raw values =================
        ax = axes[0][j]
        ax.hexbin(ls_, lu, xscale="log", yscale="log", gridsize=55,
                  cmap="viridis", mincnt=1, linewidths=0)
        ax.set_xlabel("scaled MSE  (z-scored log space)")
        ax.set_ylabel("unscaled MSE  (physical flux)")
        ax.set_title(f"{short(rd)}\nraw values: "
                     + r"$\rho_s$ = " + f"{rho:.3f}, "
                     + r"$r_p$ = " + f"{r_lin:.3f}", fontsize=10)

        # ================= BOTTOM: rank vs rank =================
        # argsort twice gives the rank of each element. This IS Spearman.
        rank_s = np.argsort(np.argsort(ls_))
        rank_u = np.argsort(np.argsort(lu))

        ax = axes[1][j]
        ax.hexbin(rank_s, rank_u, gridsize=55, cmap="magma", mincnt=1, linewidths=0)
        lim = [0, len(ls_)]
        ax.plot(lim, lim, "w--", lw=1.5, alpha=0.7, label="perfect agreement")

        # highlight the two "most anomalous" sets
        idx_s = np.array(sorted(top_s))
        ax.scatter(rank_s[idx_s], rank_u[idx_s], s=14, facecolors="none",
                   edgecolors="cyan", lw=0.7,
                   label=f"top {TOP_N} by scaled ({overlap} shared)")

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("rank by scaled MSE")
        ax.set_ylabel("rank by unscaled MSE")
        ax.set_title(r"rank vs rank — this is $\rho_s$" + f"\noverlap {overlap}/{TOP_N}",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Do the two error metrics rank spectra the same way?", fontsize=13)
    plt.tight_layout()
    fig.savefig("metric_agreement.png", dpi=150)
    print("\nwrote metric_agreement.png")

    print("""
HOW TO READ IT
  Bottom row, points hugging the diagonal -> the two metrics produce the same
  ordering, so the choice between them does not change which spectra are flagged.
  That justifies using the numerically stable log-space metric.

  Note the Pearson value is much lower than Spearman. The relationship is monotonic
  but curved (expm1), so a linear correlation understates the agreement. Since only
  the ORDER is used downstream, Spearman is the appropriate statistic.
""")


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(glob.glob("RUN_*"))
    if not runs:
        sys.exit("no RUN_* directories found — run from the folder containing them")
    main(runs)