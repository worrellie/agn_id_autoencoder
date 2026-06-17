"""
Unit tests for pure-math functions in funcs_process_gals.py.
All tests use small synthetic numpy arrays — no FITS files are read.
"""

import pytest
import numpy as np
from funcs_process_gals import deredshift_channel, calc_SNR, merge_channels


# ---------------------------------------------------------------------------
# deredshift_channel
# ---------------------------------------------------------------------------


def test_deredshift_wavelength_scale():
    """Wavelength output matches the formula l_out = l * (1 + de_z) / (1 + z)."""
    flux = np.array([1.0, 2.0, 3.0])
    l = np.array([5000.0, 5500.0, 6000.0])
    z, de_z = 0.5, 0.8

    _, l_out = deredshift_channel(flux, l, z, de_z)

    np.testing.assert_allclose(l_out, l * (1 + de_z) / (1 + z))


def test_deredshift_flux_scale():
    """Flux output matches the formula flux_out = flux * (1 + z) / (1 + de_z)."""
    flux = np.array([1.0, 2.0, 3.0])
    l = np.array([5000.0, 5500.0, 6000.0])
    z, de_z = 0.5, 0.8

    flux_out, _ = deredshift_channel(flux, l, z, de_z)

    np.testing.assert_allclose(flux_out, flux * (1 + z) / (1 + de_z))


def test_deredshift_round_trip():
    """Applying de-redshift and then its inverse recovers the original."""
    flux = np.array([1.0, 2.0, 3.0])
    l = np.array([5000.0, 5500.0, 6000.0])

    flux_out, l_out = deredshift_channel(flux, l, z=0.1, de_z=0.0)
    flux_back, l_back = deredshift_channel(flux_out, l_out, z=0.0, de_z=0.1)

    np.testing.assert_allclose(l_back, l, rtol=1e-6)
    np.testing.assert_allclose(flux_back, flux, rtol=1e-6)


# ---------------------------------------------------------------------------
# calc_SNR
# ---------------------------------------------------------------------------


def test_calc_snr_matches_formula():
    """SNR = mean_flux / std_flux for a random spectrum in the continuum window."""
    rng = np.random.default_rng(0)
    l = np.linspace(5100.0, 5800.0, 100)
    flux = rng.normal(loc=5.0, scale=1.0, size=100)

    mean_out, noise_out, snr_out = calc_SNR(flux, l)

    assert mean_out == pytest.approx(np.mean(flux), rel=1e-5)
    assert noise_out == pytest.approx(np.std(flux), rel=1e-5)
    assert snr_out == pytest.approx(np.mean(flux) / np.std(flux), rel=1e-5)


def test_calc_snr_outside_window_returns_zeros():
    """Wavelengths entirely outside 5100-5800 Å: continuum region too small → (0,0,0)."""
    l = np.linspace(3000.0, 4000.0, 100)
    flux = np.ones(100)

    mean_out, noise_out, snr_out = calc_SNR(flux, l)

    assert mean_out == 0.0
    assert noise_out == 0.0
    assert snr_out == 0.0


def test_calc_snr_no_exception_for_all_zero_flux():
    """A fully-masked (all-zero) spectrum must not raise, just return zeroes."""
    l = np.linspace(5100.0, 5800.0, 50)
    flux = np.zeros(50)

    result = calc_SNR(flux, l)  # should not raise

    assert len(result) == 3


# ---------------------------------------------------------------------------
# merge_channels
# ---------------------------------------------------------------------------


def test_merge_channels_output_length():
    """Output flux array length equals len(np.arange(0, 30000, grid_size))."""
    grid_size = 4.0
    l1 = np.arange(1000.0, 2000.0, grid_size)
    f1 = np.ones(len(l1))

    flux_out, _ = merge_channels([(f1, l1)], grid_size)

    expected_len = len(np.arange(0, 30000, grid_size))
    assert len(flux_out) == expected_len


def test_merge_channels_edge_pixels_are_zero():
    """First and last pixel of each channel are masked to 0 by merge_channels."""
    grid_size = 4.0
    l1 = np.arange(1000.0, 2000.0, grid_size)
    f1 = np.ones(len(l1))

    flux_out, _ = merge_channels([(f1, l1)], grid_size)

    start_idx = int(np.round((l1[0] - 0.0) / grid_size))
    end_idx = start_idx + len(f1) - 1

    assert flux_out[start_idx] == 0.0, "First pixel of channel should be masked"
    assert flux_out[end_idx] == 0.0, "Last pixel of channel should be masked"


def test_merge_channels_interior_pixels_nonzero():
    """Interior pixels of a channel with unit flux remain 1.0 after merging."""
    grid_size = 4.0
    l1 = np.arange(1000.0, 2000.0, grid_size)
    f1 = np.ones(len(l1))

    flux_out, _ = merge_channels([(f1, l1)], grid_size)

    start_idx = int(np.round((l1[0] - 0.0) / grid_size))
    interior = flux_out[start_idx + 1 : start_idx + len(f1) - 1]

    assert (interior == 1.0).all(), "Interior channel pixels should be unmodified"
