"""
Reconstruction error vs SNR — LINEAR axes.

Run from the directory containing the RUN_* folders:
    uv run python plot_loss_vs_snr_linear.py
    uv run python plot_loss_vs_snr_linear.py RUN_a RUN_b RUN_c
    uv run python plot_loss_vs_snr_linear.py --xmax 3 --ymax 50

Produces loss_vs_snr_linear.png, three rows per model:
  TOP    full range, linear axes
  MIDDLE zoomed to the bulk of the data (default: SNR < 2, loss < 99th pct)
  BOTTOM binned median with 16-84% band — the trend, without the scatter

WHY THE LOG VERSION IS USUALLY BETTER, AND WHY THIS ONE IS STILL WORTH HAVING.

  Your SNR distribution is severely skewed: median 0.23, 99th percentile ~5, max ~10+.
  Loss is worse — p95/median is about 19. On linear axes the bulk of the points get
  crushed into the bottom-left corner and a handful of outliers own the rest of the
  figure. That is why the log version exists, and why the power-law slope (loss ~
  SNR^-2 for pure noise) is only readable in log-log.

  But linear axes answer a question log axes hide: WHAT ARE THE ACTUAL NUMBERS?
  Where does the loss level off? At what SNR does it stop falling? Those are the
  quantities you need for choosing a sample cut, and they are far easier to read
  off a linear axis.

  Hence the zoom row: same data, restricted to the bulk, which is where any sample
  cut will actually sit.
"""

import argparse
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SNR_FLOOR = 1e-3


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
    return np.asarray(d["snr"], float), np.asarray(d["loss_unscaled"], float)


def binned(x, y, nbins=25):
    """Median and 16-84 percentile of y in equal-COUNT bins of x."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    cx, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 10:
            continue
        cx.append(np.median(x[m]))
        med.append(np.median(y[m]))
        lo.append(np.percentile(y[m], 16))
        hi.append(np.percentile(y[m], 84))
    return map(np.array, (cx, med, lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=None)
    ap.add_argument("--xmax", type=float, default=2.0,
                    help="SNR limit for the zoom row (default 2)")
    ap.add_argument("--ymax", type=float, default=None,
                    help="loss limit for the zoom row (default: 99th percentile)")
    args = ap.parse_args()

    run_dirs = args.runs or sorted(glob.glob("RUN_*"))
    if not run_dirs:
        sys.exit("no RUN_* directories found — run from the folder containing them")

    n = len(run_dirs)
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 13), squeeze=False)

    print(f"{'model':<22} {'rho':>8} {'med loss':>10} {'loss@SNR<0.5':>13} {'loss@SNR>2':>12}")
    print("-" * 70)

    for j, rd in enumerate(run_dirs):
        snr, loss = load(rd)
        k = (snr > SNR_FLOOR) & np.isfinite(loss) & (loss > 0)
        s, l = snr[k], loss[k]

        rho = spearmanr(s, l)[0]
        faint = np.median(l[s < 0.5]) if (s < 0.5).any() else np.nan
        bright = np.median(l[s > 2.0]) if (s > 2.0).any() else np.nan
        print(f"{short(rd):<22} {rho:8.3f} {np.median(l):10.3f} "
              f"{faint:13.3f} {bright:12.3f}")

        # ---------------- ROW 1: full range ----------------
        ax = axes[0][j]
        ax.hexbin(s, l, gridsize=60, cmap="viridis", mincnt=1, linewidths=0)
        ax.set_xlabel("SNR")
        ax.set_ylabel("reconstruction MSE (physical)")
        ax.set_title(f"{short(rd)} — full range\n" + r"$\rho$ = " + f"{rho:.3f}",
                     fontsize=10)

        # ---------------- ROW 2: zoomed to the bulk ----------------
        ymax = args.ymax if args.ymax is not None else np.percentile(l, 99)
        zm = (s <= args.xmax) & (l <= ymax)
        ax = axes[1][j]
        ax.hexbin(s[zm], l[zm], gridsize=60, cmap="viridis", mincnt=1, linewidths=0)
        cx, med, lo, hi = binned(s[zm], l[zm])
        ax.plot(cx, med, "r-", lw=2.5, label="binned median")
        ax.set_xlabel("SNR")
        ax.set_ylabel("reconstruction MSE (physical)")
        ax.set_title(f"zoom: SNR $\\leq$ {args.xmax:g}, MSE $\\leq$ {ymax:.1f}\n"
                     f"({zm.sum()}/{len(s)} spectra shown)", fontsize=10)
        ax.legend(fontsize=8)

        # ---------------- ROW 3: the trend alone ----------------
        ax = axes[2][j]
        cx, med, lo, hi = binned(s, l, nbins=30)
        ax.plot(cx, med, "k-", lw=2.5, label="median")
        ax.fill_between(cx, lo, hi, color="k", alpha=0.18, label="16–84%")
        ax.set_xlim(0, args.xmax)
        vis = cx <= args.xmax
        if vis.any():
            ax.set_ylim(0, np.nanmax(hi[vis]) * 1.1)
        ax.axhline(np.median(l), color="tab:blue", ls=":", lw=1.5,
                   label=f"overall median {np.median(l):.2f}")
        ax.set_xlabel("SNR")
        ax.set_ylabel("reconstruction MSE (physical)")
        ax.set_title("trend only — where does it level off?", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Reconstruction error vs SNR (linear axes)", fontsize=13)
    plt.tight_layout()
    fig.savefig("loss_vs_snr_linear.png", dpi=150)
    print("\nwrote loss_vs_snr_linear.png")

    print("""
HOW TO READ IT
  Row 3 is the one to use for choosing a sample cut. Find the SNR at which the
  median loss stops falling steeply and flattens off. Below that, loss is
  noise-dominated and the model is telling you nothing about the spectrum. Above
  it, the residual reflects genuine model error — which is where any anomaly
  signal has to live.

  With median SNR ~0.23 and only ~1200 validation spectra above SNR 1, expect the
  flattening to sit well out in the tail. That is a statement about the survey
  depth, not about the model.
""")


if __name__ == "__main__":
    main()