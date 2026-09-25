"""
residual_noise_check.py — is the autoencoder denoising, and where's the real anomaly signal?

CORE IDEA
  If the AE denoises, the per-spectrum residual (input - reconstruction) is approximately
  the NOISE it removed. So residual RMS should scale with the independently-measured noise.

      residual_RMS(i) = sqrt(per-spectrum MSE) = sqrt(loss_scaled[i])   # saved by evaluate()
      noise proxy     = NOISE  (continuum-region std, stored in the H5)

TWO OUTPUTS
  1. Spearman(residual_RMS, NOISE):
       near +1  -> residual tracks noise      -> DENOISER-consistent (poor recon is expected)
       near  0  -> residual unrelated to noise -> NOT denoising, poor recon is a real failure

  2. Noise-conditioned excess (the anomaly score that ISN'T just an SNR meter):
       within bins of similar noise, flag spectra whose RMS is far ABOVE the bin median.
       These have MORE residual than their noise explains -> structured residual -> candidates.

Reads the *_test_latent.npz that funcs.evaluate already saves; no retraining.
"""
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


def residual_noise_check(latent_npz, model_h5, noise_h5, split="test",
                         loss_key="loss_scaled", n_noise_bins=10,
                         out_prefix="denoise_check"):
    """
    latent_npz : the *_test_latent.npz from the best model (losses in model_h5 row order)
    model_h5   : the H5 the model was TRAINED on (e.g. v1) — used only for its obj_id order
    noise_h5   : an H5 that HAS the NOISE dataset (e.g. v2) — NOISE is borrowed from here

    NOISE lives in v2/v3 but not v1. v1->v2 only tightened the skip + renamed keys, so for
    any shared spectrum the noise is the same physical quantity. We therefore JOIN by obj_id
    rather than by row index — the counts differ (v2 dropped a few), so positional alignment
    would mispair spectra. v1-only spectra get no noise and are dropped from this plot.
    """
    # --- per-spectrum residual RMS from the saved losses ---
    d = np.load(latent_npz, allow_pickle=True)
    if loss_key not in d.files:
        raise KeyError(f"{loss_key} not in {latent_npz} (has {d.files})")
    loss = np.asarray(d[loss_key], float)
    rms  = np.sqrt(np.clip(loss, 0, None))          # residual RMS per spectrum
    snr  = np.asarray(d["snr"], float) if "snr" in d.files and d["snr"] is not None else None

    def _ids(h5):
        with h5py.File(h5, "r") as hf:
            return np.array([s.decode() if isinstance(s, bytes) else s
                             for s in hf[split]["obj_id"][:]])

    # obj_id order the npz losses correspond to (model's own file, shuffle=False)
    model_ids = _ids(model_h5)
    if len(model_ids) != len(rms):
        raise ValueError(f"npz has {len(rms)} rows but {model_h5}:{split} has {len(model_ids)} "
                         f"obj_ids — npz is not from this file/split")

    # NOISE borrowed from the file that has it, mapped id -> noise
    with h5py.File(noise_h5, "r") as hf:
        if "NOISE" not in hf[split]:
            raise KeyError(f"'NOISE' not in {noise_h5}:{split}")
        noise_ids = _ids(noise_h5)
        noise_vals = np.asarray(hf[split]["NOISE"][:], float)
    id2noise = dict(zip(noise_ids, noise_vals))

    # align noise onto the model's row order by id; unmatched -> nan
    noise = np.array([id2noise.get(i, np.nan) for i in model_ids])
    n_unmatched = int(np.isnan(noise).sum())
    if n_unmatched:
        print(f"{n_unmatched} spectra in {model_h5} have no match in {noise_h5} "
              f"(v1-only, dropped from this plot)")

    ok = np.isfinite(rms) & np.isfinite(noise) & (noise > 0)
    if snr is not None:
        ok &= np.isfinite(snr)
    rms_o, noise_o = rms[ok], noise[ok]
    snr_o = snr[ok] if snr is not None else None
    idx_map = np.where(ok)[0]                        # map back to original model-file test indices
    print(f"{ok.sum()} usable spectra of {len(rms)}\n")

    # --- TEST 1: what does the residual track? ---
    # For a CONTINUUM-NORMALISED model (flux / NORM_CMN), the model never sees absolute
    # flux — brightness is divided out. So the relevant noise is RELATIVE noise
    # (noise / continuum ~ 1/SNR), NOT absolute physical NOISE. Report both, but SNR is
    # the operative signature: strong -ve rho(RMS, SNR) == the residual IS the relative
    # noise == the model is denoising (equivalently, an SNR meter in normalised space).
    if spearmanr is not None:
        rho_n, p_n = spearmanr(rms_o, noise_o)
        print(f"Spearman(residual_RMS, NOISE) = {rho_n:+.3f}  (p={p_n:.1e})   "
              f"[absolute noise — WRONG yardstick for a normalised model]")
        if snr_o is not None:
            rho_s, p_s = spearmanr(rms_o, snr_o)
            print(f"Spearman(residual_RMS, SNR)   = {rho_s:+.3f}  (p={p_s:.1e})   "
                  f"[relative noise proxy — the operative one]")
            if abs(rho_s) > 0.7:
                print("  -> STRONG SNR dependence: residual IS the relative noise.")
                print("     The model is DENOISING; low variance-explained is expected,")
                print("     and raw MSE is an SNR meter — condition on SNR for anomalies.")
            elif abs(rho_s) > 0.3:
                print("  -> MODERATE: residual partly relative-noise, partly other.")
            else:
                print("  -> WEAK on SNR too: poor recon is a real failure; look for")
                print("     wavelength-structured residuals (aggregate |residual| per pixel).")
    else:
        print("scipy missing — install scipy for the correlations")

    # --- TEST 2: SNR-conditioned excess = anomaly candidates ---
    # Condition on SNR (the confound the residual actually tracks), not absolute noise.
    # Flag spectra with more residual than OTHERS AT THE SAME SNR: that excess is the
    # structured residual the noise level does NOT explain -> real anomaly candidates.
    cond = snr_o if snr_o is not None else noise_o
    cond_name = "SNR" if snr_o is not None else "NOISE"
    edges = np.quantile(cond, np.linspace(0, 1, n_noise_bins + 1))
    edges[-1] += 1e-9 * (1 + abs(edges[-1]))
    binid = np.clip(np.digitize(cond, edges) - 1, 0, n_noise_bins - 1)
    excess = np.full_like(rms_o, np.nan)
    for b in range(n_noise_bins):
        m = binid == b
        if m.sum() > 2:
            med = np.median(rms_o[m])
            mad = np.median(np.abs(rms_o[m] - med)) + 1e-9
            excess[m] = (rms_o[m] - med) / mad      # robust z within SNR bin

    order = np.argsort(np.nan_to_num(excess, nan=-np.inf))[::-1]
    print(f"\ntop 15 anomaly candidates (high residual FOR THEIR {cond_name}):")
    hdr_c = cond_name.lower()
    print(f"  {'test_idx':>8}  {'RMS':>7}  {hdr_c:>9}  {'excess_z':>8}")
    for k in order[:15]:
        print(f"  {idx_map[k]:>8}  {rms_o[k]:>7.3f}  {cond[k]:>9.4g}  {excess[k]:>8.2f}")

    # --- figure: two panels — vs NOISE (log x, reference) and vs SNR (the real signal) ---
    top = order[:15]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # panel 1: RMS vs absolute NOISE, log x (values ~1e-18)
    ax = axes[0]
    ax.scatter(noise_o, rms_o, c=np.nan_to_num(excess), s=6, alpha=0.4,
               cmap="coolwarm", vmin=-3, vmax=3, rasterized=True)
    ax.scatter(noise_o[top], rms_o[top], s=55, facecolor="none", edgecolor="k", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("NOISE (continuum-region std, physical) — log scale")
    ax.set_ylabel("residual RMS = sqrt(per-spectrum MSE), scaled")
    ax.set_title(f"vs absolute noise  (rho={rho_n:+.2f}, weak — wrong space)")

    # panel 2: RMS vs SNR — the operative relationship
    ax = axes[1]
    if snr_o is not None:
        sc = ax.scatter(snr_o, rms_o, c=np.nan_to_num(excess), s=6, alpha=0.4,
                        cmap="coolwarm", vmin=-3, vmax=3, rasterized=True)
        ax.scatter(snr_o[top], rms_o[top], s=55, facecolor="none", edgecolor="k", lw=1.2,
                   label="top anomaly candidates")
        # median RMS per SNR bin — the "expected residual given SNR" curve
        centres = 0.5 * (edges[:-1] + edges[1:])
        med_rms = [np.median(rms_o[binid == b]) if (binid == b).sum() else np.nan
                   for b in range(n_noise_bins)]
        ax.plot(centres, med_rms, "k-", lw=1.5, alpha=0.7, label="median RMS per SNR bin")
        ax.set_xscale("log")
        ax.set_xlabel("SNR (relative-noise proxy) — log scale")
        ax.set_title(f"vs SNR  (rho={rho_s:+.2f} — denoiser signature)")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label=f"excess z (residual beyond {cond_name})")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rms_vs_noise.png", dpi=150)
    print(f"\nwrote {out_prefix}_rms_vs_noise.png")

    return {"idx": idx_map, "rms": rms_o, "noise": noise_o,
            "snr": snr_o, "excess": excess}


if __name__ == "__main__":
    # losses come from the best MEAN model (trained on v1);
    # NOISE is borrowed from v2, joined by obj_id.
    residual_noise_check(
        latent_npz="/home/worrellie/Documents/agn_id_autoencoder/sae_sweep_1/RUN_StandardAutoencoder_nl1_ls256_e300_ReLU_B0e+00_lr3e-05_wd1.390913335133999e-06_esFalse_nTrue_z3ujyl6a/RUN_StandardAutoencoder_nl1_ls256_e300_ReLU_B0e+00_lr3e-05_wd1.390913335133999e-06_esFalse_nTrue_z3ujyl6a_validation_latent.npz",
        model_h5="all_spectra_float32.h5",       # v1 — the file the model was trained on
        noise_h5="all_spectra_float32_v2.h5",    # v2 — has the NOISE dataset
        split="validation",
    )
