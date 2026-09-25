"""
compare_sweeps.py — compare the mean vs median (vs SNR-variant) autoencoder sweeps.

Two levels, because "which is better" has two different answers:

  LEVEL 1 (sweep-wide, W&B):   distribution of the SCALED sweep target
                               final/best_valid_log_mse across all runs. Answers "which
                               representation reconstructs better". Scaled is comparable
                               because each flux is standardised to unit variance, so
                               MSE=1.0 is the shared "reconstructs nothing" floor and
                               (1 - MSE) is variance explained. Unscaled MSE is NOT used:
                               its expm1 de-log makes it outlier-dominated and puts the
                               two representations on different rulers.

  LEVEL 2 (best model, local): per-spectrum test losses, index-aligned on the FIXED test
                               set, for each sweep's winning model. Answers the question
                               that actually matters for anomaly detection: how much is
                               each model's reconstruction error just an SNR meter?
                               Lower |rho(loss, SNR)| = less SNR-driven = better detector.

Nothing here retrains anything — Level 2 reads the *_test_latent.npz that funcs.evaluate
already saves per run (latent/redshift/snr/loss_scaled/loss_unscaled).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None
    print("scipy not available — Level 2 rank correlations will be skipped")


# ----------------------------------------------------------------------------- #
# LEVEL 1 — sweep-wide distribution comparison from the W&B API                  #
# ----------------------------------------------------------------------------- #
def compare_sweeps_wandb(entity, project, sweeps,
                         metric="final/best_valid_log_mse",
                         out="sweep_distribution_comparison.png"):
    """
    sweeps : dict  label -> sweep_id, e.g. {"mean": "abc123", "median": "def456"}

    Compares the DISTRIBUTION of the SCALED validation MSE (the actual sweep target).
    This IS comparable across the two representations: each flux is standardised to
    unit variance against its own stats, so MSE = 1.0 means "reconstructs nothing"
    for BOTH. The value is therefore read as fraction-of-variance-unexplained, and
    (1 - MSE) as variance explained — a shared reference despite different flux.

    Deliberately does NOT use unscaled MSE: the expm1 de-log inverse makes it
    exponentially outlier-dominated and puts the two representations on different
    rulers (a smaller median-normalised range gives smaller squared errors for
    reasons of scale, not fidelity), so unscaled values are not cross-comparable.
    """
    import wandb
    api = wandb.Api()

    results = {}
    for label, sid in sweeps.items():
        sweep = api.sweep(f"{entity}/{project}/{sid}")
        rows = []
        for r in sweep.runs:
            if r.state != "finished":
                continue
            s = r.summary
            rows.append({
                "name": r.name,
                "scaled": s.get(metric),                       # the sweep target
                "latent_size": r.config.get("latent_size"),
                "learn_rate": r.config.get("learn_rate"),
                "n_layers": r.config.get("n_layers"),
                "activation": r.config.get("activation"),
            })
        vals = np.array([x["scaled"] for x in rows if x["scaled"] is not None], float)
        results[label] = {"rows": rows, "scaled": vals}
        print(f"\n== {label}  (sweep {sid}) ==")
        print(f"  finished runs with metric: {len(vals)}")
        if len(vals):
            best = min(rows, key=lambda x: (x['scaled'] if x['scaled'] is not None else np.inf))
            # scaled MSE vs the unit-variance floor: (1 - MSE) = variance explained
            print(f"  SCALED log_mse  best={vals.min():.4f}  median={np.median(vals):.4f}  "
                  f"max={vals.max():.4f}")
            print(f"  variance explained (1 - MSE):  best={1 - vals.min():.3f}  "
                  f"median={1 - np.median(vals):.3f}   [MSE=1.0 = reconstructs nothing]")
            print(f"  best config: latent={best['latent_size']} lr={best['learn_rate']:.2e} "
                  f"n_layers={best['n_layers']} act={best['activation']}")

    print("\n  Comparing SCALED log_mse: each flux is standardised to unit variance "
          "against its\n  own stats, so MSE=1.0 is the shared 'reconstructs nothing' "
          "floor for both sweeps.\n  (1 - MSE) = variance explained is the like-for-like "
          "reconstruction comparison.")

    # overlaid distributions — robustly scaled so a few diverged runs don't
    # squash every good run into one leftmost bin.
    all_vals = np.concatenate([d["scaled"] for d in results.values() if len(d["scaled"])])

    # shared x-range from a high percentile of the POOLED data, so both sweeps
    # sit on the same axis and the tail of bad runs is clipped, not plotted flat.
    lo = all_vals.min()
    hi = np.percentile(all_vals, 95)          # ignore the worst ~5% for the axis
    if hi <= lo:                               # degenerate (all near-identical)
        hi = lo + max(1e-6, 0.01 * abs(lo))
    pad = 0.03 * (hi - lo)
    x_lo, x_hi = lo - pad, hi + pad
    bins = np.linspace(x_lo, x_hi, 30)         # common bin edges for both sweeps

    fig, (ax_main, ax_box) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]})

    for label, d in results.items():
        v = d["scaled"]
        if not len(v):
            continue
        n_out = int((v > x_hi).sum())          # runs off the right edge
        lbl = f"{label} (n={len(v)}, best={v.min():.3f}, median={np.median(v):.3f}"
        lbl += f", {n_out} off-axis)" if n_out else ")"
        ax_main.hist(np.clip(v, None, x_hi), bins=bins, alpha=0.5, label=lbl)
        ax_main.axvline(v.min(), ls="--", lw=1, alpha=0.7)   # mark each sweep's best

    # 1.0 = unit-variance floor. Only draw it if it's near the plotted range,
    # otherwise it just compresses the axis again.
    if x_lo <= 1.0 <= x_hi + (x_hi - x_lo):
        ax_main.axvline(1.0, color="k", ls=":", lw=1, alpha=0.6,
                        label="MSE=1.0 (reconstructs ~nothing)")
    ax_main.set_ylabel("runs")
    ax_main.set_title("Sweep comparison — scaled validation log-MSE (sweep target; x clipped at pooled p95)")
    ax_main.legend(fontsize=8)

    # a strip of box plots underneath shows medians/IQR without any binning artefact
    labels = [l for l in results if len(results[l]["scaled"])]
    ax_box.boxplot([results[l]["scaled"] for l in labels],
                   vert=False, tick_labels=labels, widths=0.6, showfliers=False)
    ax_box.set_xlabel("final/best_valid_log_mse  (lower = better; 1.0 = reconstructs nothing)")
    ax_box.set_xlim(x_lo, x_hi)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    return results


# ----------------------------------------------------------------------------- #
# LEVEL 2 — best-model per-spectrum comparison on the fixed test set             #
# ----------------------------------------------------------------------------- #
def _load_latent_npz(path):
    """Loads a *_test_latent.npz saved by funcs.evaluate. Returns dict of arrays."""
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in ("loss_scaled", "loss_unscaled", "redshift", "snr"):
        out[k] = d[k] if k in d.files else None
    return out


def compare_best_models(model_npz, out_prefix="best_model_comparison"):
    """
    model_npz : dict  label -> path to that sweep-winner's *_test_latent.npz
                e.g. {"mean": "sae_.../sae_..._test_latent.npz", "median": "..."}

    The test split is deterministic and evaluate() runs it with shuffle=False, so row i
    is the SAME spectrum in every file — we can align by position. Asserts equal length.
    """
    data = {label: _load_latent_npz(p) for label, p in model_npz.items()}

    # --- length / alignment sanity ---
    lengths = {label: len(d["loss_unscaled"]) for label, d in data.items()
               if d["loss_unscaled"] is not None}
    if len(set(lengths.values())) != 1:
        print(f"!! test-set lengths differ across models: {lengths}\n"
              f"   these are not the same fixed test set — cannot align by index.")
        return
    n = next(iter(lengths.values()))
    print(f"aligned test set: {n} spectra\n")

    # --- the SNR-meter diagnostic, per model ---
    # For anomaly detection you WANT loss to be driven by real spectral structure,
    # not by SNR. A high |rho(loss, SNR)| means the model is mostly ranking spectra
    # by noise level — the SNR-meter failure. Compare which representation is less so.
    # Rank correlations use SCALED loss. Spearman is invariant to the expm1 de-log
    # anyway (monotonic), so scaled vs unscaled gives the same rho — but scaled keeps
    # the magnitude plots below honest, so use it throughout for consistency.
    print("== loss vs SNR / redshift  (Spearman rho; nearer 0 on SNR is better) ==")
    if spearmanr is not None:
        for label, d in data.items():
            loss = d["loss_scaled"]
            for driver in ("snr", "redshift"):
                x = d[driver]
                if x is None:
                    print(f"  {label:8s}  {driver}: (absent)")
                    continue
                rho, p = spearmanr(loss, x, nan_policy="omit")
                tag = ""
                if driver == "snr":
                    tag = "  <- strong = SNR meter" if abs(rho) > 0.5 else "  <- weak SNR dependence (good)"
                print(f"  {label:8s}  {driver:8s}: rho={rho:+.3f} (p={p:.1e}){tag}")
    else:
        print("  scipy missing — skipped")

    # --- do the two models agree on WHICH spectra are hard? ---
    # High agreement -> representation choice barely changes the ranking.
    # Low agreement  -> they surface different outliers, which is the interesting case.
    if spearmanr is not None and len(data) == 2:
        (la, da), (lb, db) = list(data.items())
        rho, p = spearmanr(da["loss_scaled"], db["loss_scaled"])
        print(f"\n== cross-model loss ranking: {la} vs {lb} ==")
        print(f"  Spearman(loss_{la}, loss_{lb}) = {rho:+.3f} (p={p:.1e})")
        print(f"  {'high — same spectra hard for both' if rho > 0.8 else 'low — they rank spectra differently'}")

    # --- figures (scaled loss: comparable, ~unit-variance, no expm1 blow-up) ---
    # (1) per-spectrum scaled loss distributions
    plt.figure(figsize=(9, 5))
    for label, d in data.items():
        l = d["loss_scaled"]
        plt.hist(l, bins=60, alpha=0.5, label=f"{label} (median={np.median(l):.4f})")
    plt.xlabel("per-spectrum scaled test MSE  (1.0 = reconstructs nothing)")
    plt.ylabel("N")
    plt.yscale("log")   # anomaly tail lives in the log scale
    plt.title("Best-model per-spectrum loss distribution (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss_hist.png", dpi=150)

    # (2) loss vs SNR scatter, one panel per model
    have_snr = {k: v for k, v in data.items() if v["snr"] is not None}
    if have_snr:
        fig, axes = plt.subplots(1, len(have_snr), figsize=(7 * len(have_snr), 5), squeeze=False)
        for ax, (label, d) in zip(axes[0], have_snr.items()):
            ax.scatter(d["snr"], d["loss_scaled"], s=3, alpha=0.3, rasterized=True)
            ax.set_xlabel("SNR")
            ax.set_ylabel("scaled test MSE")
            ax.set_yscale("log")
            ax.set_title(f"{label}: loss vs SNR")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_loss_vs_snr.png", dpi=150)

    print(f"\n  wrote {out_prefix}_loss_hist.png"
          + ("" if not have_snr else f" and {out_prefix}_loss_vs_snr.png"))
    return data



# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    ENTITY, PROJECT = "worrellie-iastro", "agn_id_autoencoder"

    # ---- LEVEL 1: fill in the two sweep IDs ----
    SWEEPS = {
        "mean":   "qkh4r13j",
        "median": "nsapapun",
    }
    compare_sweeps_wandb(ENTITY, PROJECT, SWEEPS)

    # ---- LEVEL 2: point at each winner's *_test_latent.npz ----
    # find the winning run in the W&B UI (lowest best_valid_unscaled_mse),
    # then its local test_name dir holds {test_name}_test_latent.npz
    MODELS = {
        "mean":   "<mean_best_test_name>/<mean_best_test_name>_test_latent.npz",
        "median": "<median_best_test_name>/<median_best_test_name>_test_latent.npz",
    }
    # compare_best_models(MODELS)

    print("Fill in SWEEPS / MODELS above, then uncomment the call(s) you want.")