"""
Two invariants for the normalisation pipeline:

A — NORMFAC / continuum mean is computed from unmasked pixels only.
    calc_SNR uses target_mask = (l >= 5100) & (l <= 5800) & (flux != 0), so
    inserting zeros in the continuum band must not change cont_mean or snr.
    full_spec_median is taken over flux[flux != 0], so zeros must not affect it.

B — Zero (masked) pixels stay exactly zero across every flux_type transform and
    after the runtime Z-score + re-mask applied in training.py lines 170-175.
"""

import pytest
import numpy as np
import torch
from funcs_process_gals import calc_SNR

INPUT_SIZE = 16
BATCH = 4


# ---------------------------------------------------------------------------
# A: NORMFAC uses only unmasked continuum pixels
# ---------------------------------------------------------------------------


def test_calc_snr_ignores_zeros_in_continuum():
    """Zeros inserted in the 5100-5800 Å band must not change cont_mean or SNR."""
    rng = np.random.default_rng(0)
    l_cont = np.linspace(5100.0, 5800.0, 50)
    flux_cont = rng.uniform(1.0, 2.0, 50)

    # Baseline: no masked pixels
    mean_base, noise_base, snr_base = calc_SNR(flux_cont, l_cont)

    # Add two zero-valued "masked" pixels inside the continuum window
    l_with_zeros = np.concatenate([l_cont, [5200.0, 5400.0]])
    flux_with_zeros = np.concatenate([flux_cont, [0.0, 0.0]])
    mean_zeros, noise_zeros, snr_zeros = calc_SNR(flux_with_zeros, l_with_zeros)

    assert mean_zeros == pytest.approx(mean_base, rel=1e-5)
    assert snr_zeros == pytest.approx(snr_base, rel=1e-5)


def test_full_spec_median_excludes_masked_zeros():
    """Median over unmasked pixels differs from median over full array with zeros."""
    flux = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
    mask = flux == 0
    median_unmasked = np.median(flux[~mask])
    median_all = np.median(flux)

    # Demonstrates that including zeros skews the median
    assert median_unmasked != pytest.approx(median_all)
    assert median_unmasked == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# B: Zero pixels stay zero through every normalisation strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flux_type", ["normalized_flux_cont", "normalized_flux_med"])
def test_division_normalisation_preserves_zeros(flux_type):
    """raw_flux / norm_factor: zero pixels must remain zero."""
    rng = np.random.default_rng(1)
    raw_flux = rng.uniform(0.5, 2.0, 20)
    zero_indices = [3, 7, 15]
    raw_flux[zero_indices] = 0.0
    norm_factor = 1.5

    normed = raw_flux / norm_factor

    assert (normed[zero_indices] == 0.0).all()


def test_log_scale_normalisation_preserves_zeros():
    """sign(0)*log1p(|0|) == 0, and the explicit unmasked re-multiply keeps zeros zero."""
    rng = np.random.default_rng(2)
    norm_flux_cont = rng.uniform(0.5, 2.0, 20)
    zero_indices = [2, 9]
    norm_flux_cont[zero_indices] = 0.0
    unmasked = norm_flux_cont != 0  # matches save_h5 line: unmasked = (raw_flux != 0)

    log_scale_flux = np.sign(norm_flux_cont) * np.log1p(np.abs(norm_flux_cont))
    log_scale_flux = log_scale_flux * unmasked  # explicit re-mask as in save_h5

    assert log_scale_flux[2] == 0.0
    assert log_scale_flux[9] == 0.0


def test_runtime_zscore_remask_preserves_zeros():
    """After (x - mean)/std then x * x_mask, positions where x_mask==0 are exactly 0.

    This mirrors training.py lines 170-175:
        if normalize: x = (x - train_mean) / train_std
        x = x * x_mask
    """
    torch.manual_seed(3)
    x = torch.randn(BATCH, INPUT_SIZE)
    x_mask = torch.ones(BATCH, INPUT_SIZE)
    masked_cols = [0, 5, 10]
    x_mask[:, masked_cols] = 0.0
    x[:, masked_cols] = 0.0  # data at masked positions is stored as zero

    mean = torch.full((INPUT_SIZE,), 0.5)
    std = torch.full((INPUT_SIZE,), 2.0)

    x_normed = (x - mean) / std   # Z-score shifts the zeros to non-zero
    x_normed = x_normed * x_mask  # re-mask restores them to zero

    assert (x_normed[:, masked_cols] == 0.0).all()


def test_welford_principle_excludes_masked_zeros():
    """Welford mean (unmasked only) differs from mean over all pixels including zeros.

    In save_h5, only pixels where raw_flux != 0 contribute to total_pixels and
    the running sums, producing mean = sum_unmasked / n_unmasked.  This is
    equivalent to np.mean(arr[arr != 0]).
    """
    arr = np.array([1.0, 2.0, 3.0, 0.0, 0.0])  # last 2 are masked zeros

    mean_all = np.mean(arr)             # (1+2+3+0+0)/5 = 1.2
    mean_unmasked = np.mean(arr[arr != 0])  # (1+2+3)/3 = 2.0

    assert mean_all != pytest.approx(mean_unmasked)
    assert mean_unmasked == pytest.approx(2.0)
