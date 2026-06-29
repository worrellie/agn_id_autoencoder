"""
Test suite for process_spectra.py

All tests are fully self-contained: no real FITS, JSON, or catalogue files
are read from disk. Every file I/O call is replaced with a unittest.mock
patch or a fixture-supplied numpy array.

Organised into one TestCase class per public function, in the same order
they appear in the source file.
"""

import json
import os
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pytest

import process_spectra
from process_spectra import (
    calc_SNR,
    check_common_region_exists,
    crop_spectrum,
    deredshift_channel,
    get_channel_data,
    get_common_grid,
    get_id,
    get_valid_triplets,
    initialize_worker,
    load_common_region,
    make_col_name_fits_compatible,
    merge_channels,
    merge_orignal_de_z,
    save_spec,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_flux():
    """100-element uniform flux array (1e-17 erg cm-2 s-1 AA-1)."""
    return np.ones(100) * 1e-17


@pytest.fixture
def simple_wavelength():
    """100-element wavelength array from 5000 to 5400 Å."""
    return np.linspace(5000.0, 5400.0, 100)


@pytest.fixture
def snr_wavelengths():
    """500-element array spanning 5000-5900 Å, covering the SNR target 5100-5800."""
    return np.linspace(5000.0, 5900.0, 500)


@pytest.fixture
def snr_flux_noisy(snr_wavelengths):
    """Non-uniform flux across snr_wavelengths so that std > 0."""
    rng = np.random.default_rng(42)
    return 1e-17 + 1e-19 * rng.standard_normal(len(snr_wavelengths))


# grid_size used by all merge_channels fixtures
_GRID = 4.0
# Channel 1 occupies indices 1500-1749 of full_l (6000 / 4 = 1500)
_L1 = np.arange(6000.0, 7000.0, _GRID)  # 250 pixels, end_idx = 1750
_F1 = np.full(len(_L1), 1e-17)


@pytest.fixture
def channel_pairs_normal():
    """Two channels with a 2-pixel expected gap (overlap in [−2, 2])."""
    # start_idx of ch2 = 7008/4 = 1752  →  gap = 1752 − 1750 = 2
    l2 = np.arange(7008.0, 8000.0, _GRID)
    f2 = np.full(len(l2), 2e-17)
    return [[_F1.copy(), _L1.copy()], [f2, l2]]


@pytest.fixture
def channel_pairs_instrument_gap():
    """Two channels separated by a 150-pixel instrument gap (100 ≤ gap ≤ 300)."""
    # start_idx of ch2 = 7600/4 = 1900  →  gap = 1900 − 1750 = 150
    l2 = np.arange(7600.0, 8500.0, _GRID)
    f2 = np.full(len(l2), 2e-17)
    return [[_F1.copy(), _L1.copy()], [f2, l2]]


@pytest.fixture
def channel_pairs_unexpected_gap():
    """Two channels separated by a 500-pixel gap (outside any expected range)."""
    # start_idx of ch2 = 9000/4 = 2250  →  gap = 2250 − 1750 = 500
    l2 = np.arange(9000.0, 10000.0, _GRID)
    f2 = np.full(len(l2), 2e-17)
    return [[_F1.copy(), _L1.copy()], [f2, l2]]


@pytest.fixture
def common_vals():
    """Wavelength crop window for crop_spectrum tests."""
    return [6500.0, 18000.0]


# ---------------------------------------------------------------------------
# 1. deredshift_channel
# ---------------------------------------------------------------------------


class TestDeredshiftChannel:

    def test_wavelength_formula(self, simple_flux, simple_wavelength):
        z, de_z = 1.2, 0.8
        _, l_z = deredshift_channel(simple_flux, simple_wavelength, z, de_z=de_z)
        expected = (1 + de_z) * simple_wavelength / (1 + z)
        assert np.allclose(l_z, expected)

    def test_wavelength_shorter_when_z_gt_de_z(self, simple_flux, simple_wavelength):
        # Deredshifting towards a lower frame compresses the wavelength axis
        _, l_z = deredshift_channel(simple_flux, simple_wavelength, z=1.5, de_z=0.9)
        assert np.all(l_z < simple_wavelength)

    def test_flux_formula(self, simple_flux, simple_wavelength):
        z, de_z = 1.2, 0.8
        flux_z, _ = deredshift_channel(simple_flux, simple_wavelength, z, de_z=de_z)
        expected = simple_flux * (1 + z) / (1 + de_z)
        assert np.allclose(flux_z, expected)

    def test_shape_preserved(self, simple_flux, simple_wavelength):
        flux_z, l_z = deredshift_channel(simple_flux, simple_wavelength, z=1.0, de_z=0.9)
        assert flux_z.shape == simple_flux.shape
        assert l_z.shape == simple_wavelength.shape

    def test_identity_when_z_equals_de_z(self, simple_flux, simple_wavelength):
        # When observed redshift == target frame, both arrays must be unchanged
        z = 0.9
        flux_z, l_z = deredshift_channel(simple_flux, simple_wavelength, z=z, de_z=z)
        assert np.allclose(flux_z, simple_flux)
        assert np.allclose(l_z, simple_wavelength)


# ---------------------------------------------------------------------------
# 2. crop_spectrum
# ---------------------------------------------------------------------------


class TestCropSpectrum:

    def test_all_returned_wavelengths_within_range(self, simple_flux, simple_wavelength):
        lo, hi = 5100.0, 5300.0
        _, l_out = crop_spectrum(simple_flux, simple_wavelength, [lo, hi])
        assert np.all(l_out >= np.ceil(lo))
        assert np.all(l_out <= np.floor(hi))

    def test_flux_and_wavelength_lengths_match(self, simple_flux, simple_wavelength):
        f_out, l_out = crop_spectrum(simple_flux, simple_wavelength, [5100.0, 5300.0])
        assert len(f_out) == len(l_out)

    def test_window_wider_than_array_returns_all(self, simple_flux, simple_wavelength):
        f_out, l_out = crop_spectrum(simple_flux, simple_wavelength, [4000.0, 6000.0])
        assert len(f_out) == len(simple_flux)

    def test_window_misses_array_returns_empty(self, simple_flux, simple_wavelength):
        f_out, l_out = crop_spectrum(simple_flux, simple_wavelength, [9000.0, 10000.0])
        assert len(f_out) == 0
        assert len(l_out) == 0

    def test_flux_values_are_drawn_from_original(self, simple_wavelength):
        flux = np.arange(100, dtype=float)
        f_out, _ = crop_spectrum(flux, simple_wavelength, [5100.0, 5300.0])
        assert set(f_out.tolist()).issubset(set(flux.tolist()))


# ---------------------------------------------------------------------------
# 3. merge_channels
# ---------------------------------------------------------------------------


class TestMergeChannels:

    FULL_LEN = len(np.arange(0, 30000, _GRID))

    def test_output_length_equals_full_grid(self, channel_pairs_normal):
        full_flux, _ = merge_channels(channel_pairs_normal, _GRID)
        assert len(full_flux) == self.FULL_LEN

    def test_interior_flux_placed_at_correct_index(self):
        # 5-pixel channel starting at 6000 Å → start_idx = 6000/4 = 1500
        l1 = np.arange(6000.0, 6020.0, _GRID)
        f1 = np.full(len(l1), 5e-17)
        full_flux, _ = merge_channels([[f1, l1]], _GRID)
        start_idx = int(round(6000.0 / _GRID))  # 1500
        # Interior pixels (not first or last) hold the original flux value
        assert full_flux[start_idx + 1] == pytest.approx(5e-17)
        assert full_flux[start_idx + 2] == pytest.approx(5e-17)

    def test_edge_pixels_masked_to_zero(self):
        # The first and last pixel of every channel block are zeroed by the mask step
        l1 = np.arange(6000.0, 6020.0, _GRID)
        f1 = np.full(len(l1), 5e-17)
        full_flux, _ = merge_channels([[f1, l1]], _GRID)
        start_idx = int(round(6000.0 / _GRID))
        end_idx = start_idx + len(f1)
        assert full_flux[start_idx] == 0.0
        assert full_flux[end_idx - 1] == 0.0

    def test_expected_pixel_gap_no_warning(self, channel_pairs_normal, capsys):
        merge_channels(channel_pairs_normal, _GRID)
        assert "unexpected gap" not in capsys.readouterr().out

    def test_instrument_gap_no_warning(self, channel_pairs_instrument_gap, capsys):
        merge_channels(channel_pairs_instrument_gap, _GRID)
        assert "unexpected gap" not in capsys.readouterr().out

    def test_unexpected_gap_prints_warning(self, channel_pairs_unexpected_gap, capsys):
        merge_channels(channel_pairs_unexpected_gap, _GRID)
        assert "unexpected gap!!" in capsys.readouterr().out

    def test_second_channel_overwrites_its_region(self):
        # Two non-overlapping channels; interior of ch2 must hold ch2's flux, not zeros
        l1 = np.arange(6000.0, 6020.0, _GRID)
        l2 = np.arange(7000.0, 7020.0, _GRID)
        f1 = np.full(len(l1), 1e-17)
        f2 = np.full(len(l2), 9e-17)
        full_flux, _ = merge_channels([[f1, l1], [f2, l2]], _GRID)
        idx2 = int(round(7000.0 / _GRID))
        assert full_flux[idx2 + 1] == pytest.approx(9e-17)


# ---------------------------------------------------------------------------
# 4. calc_SNR
# ---------------------------------------------------------------------------


class TestCalcSNR:

    def test_normal_case_returns_positive_snr(self, snr_flux_noisy, snr_wavelengths):
        mean, noise, snr = calc_SNR(snr_flux_noisy, snr_wavelengths)
        assert mean > 0
        assert noise > 0
        assert snr > 0

    def test_snr_equals_mean_over_noise(self, snr_flux_noisy, snr_wavelengths):
        mean, noise, snr = calc_SNR(snr_flux_noisy, snr_wavelengths)
        assert snr == pytest.approx(mean / noise)

    def test_zero_noise_returns_mean_and_zero_snr(self):
        # Perfectly uniform flux → std == 0; function returns (mean, 0.0, 0.0)
        l = np.linspace(5100.0, 5800.0, 100)
        flux = np.full(100, 5e-17)
        mean, noise, snr = calc_SNR(flux, l)
        assert noise == 0.0
        assert snr == 0.0
        assert mean == pytest.approx(5e-17)

    def test_no_wavelengths_in_target_region(self):
        # All wavelengths below 5100 Å → target_flux.size == 0 < 2
        l = np.linspace(3000.0, 4000.0, 200)
        flux = np.ones(200) * 1e-17
        assert calc_SNR(flux, l) == (0.0, 0.0, 0.0)

    def test_only_one_wavelength_in_region(self):
        # A single-element target_flux cannot have std → treated as too small
        assert calc_SNR(np.array([1e-17]), np.array([5500.0])) == (0.0, 0.0, 0.0)

    def test_all_flux_zero_in_region_excluded(self):
        # flux != 0 mask removes all zeros → target_flux.size == 0 < 2
        l = np.linspace(5100.0, 5800.0, 200)
        flux = np.zeros(200)
        assert calc_SNR(flux, l) == (0.0, 0.0, 0.0)

    def test_only_nonzero_pixels_contribute(self):
        # Pixels outside 5100-5800 are zero; SNR is computed on the nonzero region only
        l = np.linspace(5000.0, 5900.0, 500)
        flux = np.zeros(500)
        in_region = (l >= 5100) & (l <= 5800)
        flux[in_region] = 3e-17  # uniform within region → noise == 0
        mean, noise, snr = calc_SNR(flux, l)
        assert mean == pytest.approx(3e-17)
        assert noise == 0.0


# ---------------------------------------------------------------------------
# 5. get_id
# ---------------------------------------------------------------------------


class TestGetId:

    def test_extracts_id_from_standard_base_name(self):
        assert get_id("cosmos_bagpipes_202598_2h_z1.2546") == "202598"

    def test_extracts_different_id(self):
        assert get_id("cosmos_bagpipes_999_2h_z0.95") == "999"


# ---------------------------------------------------------------------------
# 6. make_col_name_fits_compatible
# ---------------------------------------------------------------------------


class TestMakeColNameFitsCompatible:

    def test_truncates_to_seven_characters(self):
        result = make_col_name_fits_compatible("dblplaw:alpha_50")
        assert len(result) <= 7

    def test_output_is_uppercase(self):
        result = make_col_name_fits_compatible("dblplaw:alpha_50")
        assert result == result.upper()

    def test_dblplaw_prefix_removed(self):
        result = make_col_name_fits_compatible("dblplaw:alpha_50")
        assert "DBLPLAW" not in result

    def test_colon_replaced_with_underscore(self):
        result = make_col_name_fits_compatible("some:col_50")
        assert ":" not in result

    def test_dot_replaced_with_underscore(self):
        result = make_col_name_fits_compatible("some.col_50")
        assert "." not in result

    def test_target_string_removed(self):
        result = make_col_name_fits_compatible("TARGET_REDSHIFT")
        assert "TARGET" not in result

    def test_50_suffix_removed(self):
        result = make_col_name_fits_compatible("dblplaw:alpha_50")
        assert "_50" not in result


# ---------------------------------------------------------------------------
# 7. merge_orignal_de_z
# ---------------------------------------------------------------------------


class TestMergeOriginalDeZ:

    def test_wavelength_sorted_ascending(self):
        l = np.array([5300.0, 5100.0, 5200.0])
        flux = np.array([3.0, 1.0, 2.0])
        _, l_out = merge_orignal_de_z(flux, l)
        assert list(l_out) == [5100.0, 5200.0, 5300.0]

    def test_flux_reordered_with_wavelength(self):
        l = np.array([5300.0, 5100.0, 5200.0])
        flux = np.array([3.0, 1.0, 2.0])
        f_out, _ = merge_orignal_de_z(flux, l)
        assert list(f_out) == [1.0, 2.0, 3.0]

    def test_returns_list_of_two_elements(self, simple_flux, simple_wavelength):
        result = merge_orignal_de_z(simple_flux, simple_wavelength)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_already_sorted_input_unchanged(self, simple_flux, simple_wavelength):
        _, l_out = merge_orignal_de_z(simple_flux, simple_wavelength)
        assert np.allclose(l_out, simple_wavelength)


# ---------------------------------------------------------------------------
# 8. get_valid_triplets  (generator — all filesystem calls mocked)
# ---------------------------------------------------------------------------

# Constants used across the generator tests
_SPEC_DIR = "spectra"
_Z_DIR = "z1.2"
_GALAXY_Z = "1.2"
_BASE = "cosmos_bagpipes_123"
_RI_FILE = f"{_BASE}_z{_GALAXY_Z}_RI.fits"
_YJ_FILE = f"{_BASE}_z{_GALAXY_Z}_YJ.fits"
_H_FILE = f"{_BASE}_z{_GALAXY_Z}_H.fits"
_RI_PATH = os.path.join(_SPEC_DIR, _Z_DIR, _RI_FILE)
_YJ_PATH = os.path.join(_SPEC_DIR, _Z_DIR, _YJ_FILE)
_H_PATH = os.path.join(_SPEC_DIR, _Z_DIR, _H_FILE)


def _listdir_standard(path):
    """Simulates a single valid z-dir plus one non-directory entry."""
    mapping = {
        _SPEC_DIR: [_Z_DIR, "not_a_dir.txt"],
        os.path.join(_SPEC_DIR, _Z_DIR): [_RI_FILE, _YJ_FILE, _H_FILE],
    }
    return mapping.get(path, [])


def _isdir_standard(path):
    """Only the z_dir path is a directory."""
    return path == os.path.join(_SPEC_DIR, _Z_DIR)


def _glob_standard(pattern):
    """Returns the correct path for each band pattern."""
    for band, band_path in [("RI", _RI_PATH), ("YJ", _YJ_PATH), ("H", _H_PATH)]:
        if f"_{band}.fits" in pattern:
            return [band_path]
    return []


class TestGetValidTriplets:

    def test_yields_correct_triplet_paths(self):
        with patch("process_spectra.os.listdir", side_effect=_listdir_standard), \
             patch("process_spectra.os.path.isdir", side_effect=_isdir_standard), \
             patch("process_spectra.glob.glob", side_effect=_glob_standard):
            results = list(get_valid_triplets(_SPEC_DIR))

        assert len(results) == 1
        triplet, _ = results[0]
        assert triplet == [_RI_PATH, _YJ_PATH, _H_PATH]

    def test_yields_float_redshift(self):
        with patch("process_spectra.os.listdir", side_effect=_listdir_standard), \
             patch("process_spectra.os.path.isdir", side_effect=_isdir_standard), \
             patch("process_spectra.glob.glob", side_effect=_glob_standard):
            results = list(get_valid_triplets(_SPEC_DIR))

        _, z = results[0]
        assert isinstance(z, float)
        assert z == pytest.approx(1.2)

    def test_non_directory_entries_are_skipped(self):
        # When isdir always returns False, no triplets should be yielded
        with patch("process_spectra.os.listdir", side_effect=_listdir_standard), \
             patch("process_spectra.os.path.isdir", return_value=False), \
             patch("process_spectra.glob.glob", side_effect=_glob_standard):
            results = list(get_valid_triplets(_SPEC_DIR))

        assert results == []

    def test_files_without_cosmos_bagpipes_prefix_are_skipped(self):
        def listdir_no_match(path):
            if path == _SPEC_DIR:
                return [_Z_DIR]
            return ["other_instrument_z1.2_RI.fits"]

        with patch("process_spectra.os.listdir", side_effect=listdir_no_match), \
             patch("process_spectra.os.path.isdir", return_value=True), \
             patch("process_spectra.glob.glob", side_effect=_glob_standard):
            results = list(get_valid_triplets(_SPEC_DIR))

        assert results == []

    def test_triplet_skipped_when_band_file_missing(self):
        # Returning [] for YJ means the triplet is incomplete and must not be yielded
        def glob_missing_yj(pattern):
            return [] if "_YJ.fits" in pattern else _glob_standard(pattern)

        with patch("process_spectra.os.listdir", side_effect=_listdir_standard), \
             patch("process_spectra.os.path.isdir", side_effect=_isdir_standard), \
             patch("process_spectra.glob.glob", side_effect=glob_missing_yj):
            results = list(get_valid_triplets(_SPEC_DIR))

        assert results == []

    def test_duplicate_base_name_yields_only_one_triplet(self, capsys):
        # Three files all share the same base_name → only one triplet should be yielded
        def listdir_dup(path):
            if path == _SPEC_DIR:
                return [_Z_DIR]
            # RI file appears twice with identical name, plus YJ and H
            return [_RI_FILE, _RI_FILE, _YJ_FILE, _H_FILE]

        with patch("process_spectra.os.listdir", side_effect=listdir_dup), \
             patch("process_spectra.os.path.isdir", side_effect=_isdir_standard), \
             patch("process_spectra.glob.glob", side_effect=_glob_standard):
            results = list(get_valid_triplets(_SPEC_DIR))

        assert len(results) == 1

    def test_multiple_galaxies_yield_multiple_triplets(self):
        base2 = "cosmos_bagpipes_456"
        ri2 = f"{base2}_z{_GALAXY_Z}_RI.fits"
        yj2 = f"{base2}_z{_GALAXY_Z}_YJ.fits"
        h2 = f"{base2}_z{_GALAXY_Z}_H.fits"

        def listdir_two(path):
            if path == _SPEC_DIR:
                return [_Z_DIR]
            return [_RI_FILE, _YJ_FILE, _H_FILE, ri2, yj2, h2]

        def glob_two(pattern):
            for band in ["RI", "YJ", "H"]:
                if f"_{band}.fits" in pattern:
                    if _BASE in pattern:
                        return [os.path.join(_SPEC_DIR, _Z_DIR, f"{_BASE}_z{_GALAXY_Z}_{band}.fits")]
                    if base2 in pattern:
                        return [os.path.join(_SPEC_DIR, _Z_DIR, f"{base2}_z{_GALAXY_Z}_{band}.fits")]
            return []

        with patch("process_spectra.os.listdir", side_effect=listdir_two), \
             patch("process_spectra.os.path.isdir", side_effect=_isdir_standard), \
             patch("process_spectra.glob.glob", side_effect=glob_two):
            results = list(get_valid_triplets(_SPEC_DIR))

        assert len(results) == 2


# ---------------------------------------------------------------------------
# 9. get_channel_data  (mocks fits.getdata)
# ---------------------------------------------------------------------------


class TestGetChannelData:

    def _make_getdata_mock(self, flux_arr, templ_arr, wave_arr, t=1):
        """Returns a side_effect function for fits.getdata keyed by ext."""
        ext_map = {t: flux_arr, 4: templ_arr, 9: wave_arr}
        return lambda path, ext: ext_map[ext]

    def test_returns_correct_three_arrays(self):
        flux_arr = np.array([1.0, 2.0])
        templ_arr = np.array([3.0, 4.0])
        wave_arr = np.array([7000.0, 7004.0])

        with patch("process_spectra.fits.getdata",
                   side_effect=self._make_getdata_mock(flux_arr, templ_arr, wave_arr)):
            flux, l, f_templ = get_channel_data("fake.fits", t=1)

        np.testing.assert_array_equal(flux, flux_arr)
        np.testing.assert_array_equal(l, wave_arr)
        np.testing.assert_array_equal(f_templ, templ_arr)

    def test_custom_t_uses_correct_extension(self):
        flux_arr = np.array([10.0])
        templ_arr = np.array([20.0])
        wave_arr = np.array([8000.0])

        with patch("process_spectra.fits.getdata",
                   side_effect=self._make_getdata_mock(flux_arr, templ_arr, wave_arr, t=2)) as mock_gd:
            get_channel_data("fake.fits", t=2)

        # ext is always passed as a keyword arg in get_channel_data
        called_exts = [c.kwargs["ext"] for c in mock_gd.call_args_list]
        assert 2 in called_exts
        assert 1 not in called_exts


# ---------------------------------------------------------------------------
# 10. initialize_worker  (global _CAT mutation)
# ---------------------------------------------------------------------------


class TestInitializeWorker:

    @pytest.fixture(autouse=True)
    def reset_global_cat(self):
        """Restore the module-level _CAT to None after every test."""
        yield
        process_spectra._CAT = None

    def test_single_row_populates_cat(self):
        mock_table = [{"TARGET_ID": "999", "TARGET_REDSHIFT": 1.1, "TARGET_HMAG": 22.5}]
        with patch("process_spectra.Table.read", return_value=mock_table):
            initialize_worker("fake.fits", "TARGET_ID", ["TARGET_REDSHIFT", "TARGET_HMAG"])

        assert process_spectra._CAT == {
            "999": {"TARGET_REDSHIFT": 1.1, "TARGET_HMAG": 22.5}
        }

    def test_multiple_rows_populate_multiple_entries(self):
        mock_table = [
            {"TARGET_ID": "1", "FLUX": 1.0},
            {"TARGET_ID": "2", "FLUX": 2.0},
        ]
        with patch("process_spectra.Table.read", return_value=mock_table):
            initialize_worker("fake.fits", "TARGET_ID", ["FLUX"])

        assert len(process_spectra._CAT) == 2
        assert process_spectra._CAT["1"] == {"FLUX": 1.0}
        assert process_spectra._CAT["2"] == {"FLUX": 2.0}

    def test_numeric_id_is_converted_to_string_key(self):
        mock_table = [{"TARGET_ID": 42, "FLUX": 3.0}]
        with patch("process_spectra.Table.read", return_value=mock_table):
            initialize_worker("fake.fits", "TARGET_ID", ["FLUX"])

        # The str() conversion in the dict comprehension must produce a string key
        assert "42" in process_spectra._CAT
        assert 42 not in process_spectra._CAT

    def test_table_read_called_with_provided_path(self):
        with patch("process_spectra.Table.read", return_value=[]) as mock_read:
            initialize_worker("my_catalogue.fits", "ID", [])

        mock_read.assert_called_once_with("my_catalogue.fits")


# ---------------------------------------------------------------------------
# 11. save_spec  (mocks fits BinTableHDU write path only)
# ---------------------------------------------------------------------------


class TestSaveSpec:
    """
    fits.Column and fits.Header are real astropy objects (no I/O).
    Only fits.BinTableHDU.from_columns is mocked so writeto is never called
    on disk and the real header object can be inspected.
    """

    @pytest.fixture
    def spec_args(self):
        """Minimal valid arguments for save_spec."""
        flux = np.linspace(1e-17, 2e-17, 50)
        l = np.linspace(6500.0, 18000.0, 50)
        norm_factors = {"continuum_mean": 1.5e-17, "full_spec_median": 1.2e-17}
        ref_cat_row = {"TARGET_REDSHIFT": 1.1}
        return (flux, l, 1.1, 15.0, norm_factors, ref_cat_row,
                "cosmos_bagpipes_123_2h_z1.1", "/fake/outdir")

    def test_writeto_called_exactly_once(self, spec_args):
        mock_hdu = MagicMock()
        with patch("process_spectra.fits.BinTableHDU.from_columns", return_value=mock_hdu):
            save_spec(*spec_args)

        mock_hdu.writeto.assert_called_once()

    def test_writeto_uses_overwrite_true(self, spec_args):
        mock_hdu = MagicMock()
        with patch("process_spectra.fits.BinTableHDU.from_columns", return_value=mock_hdu):
            save_spec(*spec_args)

        assert mock_hdu.writeto.call_args.kwargs.get("overwrite") is True

    def test_output_filename_has_correct_suffix(self, spec_args):
        mock_hdu = MagicMock()
        with patch("process_spectra.fits.BinTableHDU.from_columns", return_value=mock_hdu):
            save_spec(*spec_args)

        path_arg = mock_hdu.writeto.call_args.args[0]
        assert path_arg.endswith("_noisy_deZ_rebinned.fits")

    def test_standard_header_keys_are_set(self, spec_args):
        """OG_Z, SNR, ORIGINAL, NORM_CON and NORM_MED must appear in the header."""
        captured = []

        def capture(cols, header=None):
            captured.append(header)
            return MagicMock()

        with patch("process_spectra.fits.BinTableHDU.from_columns", side_effect=capture):
            save_spec(*spec_args)

        hdr = captured[0]
        for key in ("OG_Z", "SNR", "ORIGINAL", "NORM_CON", "NORM_MED"):
            assert key in hdr, f"Expected key '{key}' in FITS header"

    def test_ref_cat_columns_written_to_header(self, spec_args):
        """Each key in ref_cat_row must appear in the header (after name mangling)."""
        captured = []

        def capture(cols, header=None):
            captured.append(header)
            return MagicMock()

        with patch("process_spectra.fits.BinTableHDU.from_columns", side_effect=capture):
            save_spec(*spec_args)

        hdr = captured[0]
        # "TARGET_REDSHIFT" is mangled by make_col_name_fits_compatible
        expected_key = make_col_name_fits_compatible("TARGET_REDSHIFT")
        assert expected_key in hdr


# ---------------------------------------------------------------------------
# 12. load_common_region  (mocks builtins.open)
# ---------------------------------------------------------------------------


class TestLoadCommonRegion:

    def test_returns_dict(self):
        region = {"common_min": 6500.0, "common_max": 18000.0}
        with patch("builtins.open", mock_open(read_data=json.dumps(region))):
            result = load_common_region("common_region.json")

        assert isinstance(result, dict)

    def test_parses_all_json_values(self):
        region = {"common_min": 6500.0, "common_max": 18000.0, "z_target": 0.9}
        with patch("builtins.open", mock_open(read_data=json.dumps(region))):
            result = load_common_region("common_region.json")

        assert result["common_min"] == 6500.0
        assert result["z_target"] == 0.9


# ---------------------------------------------------------------------------
# 13. check_common_region_exists  (mocks os.path.exists)
# ---------------------------------------------------------------------------


class TestCheckCommonRegionExists:

    def test_returns_true_when_file_exists(self):
        with patch("process_spectra.os.path.exists", return_value=True):
            assert check_common_region_exists("common_region.json") is True

    def test_returns_false_when_file_missing(self):
        with patch("process_spectra.os.path.exists", return_value=False):
            assert check_common_region_exists("common_region.json") is False


# ---------------------------------------------------------------------------
# 14. get_common_grid  (mocks fits.getheader, Table.read, file I/O)
# ---------------------------------------------------------------------------


class TestGetCommonGrid:

    # Three-band reference triplet used across all grid tests
    REF_TRIPLET = {
        "RI": "fake_RI.fits",
        "YJ": "fake_YJ.fits",
        "H": "fake_H.fits",
    }

    # Header returned for every band: obs window 7000–25000 Å
    MOCK_HEADER = {"WMIN": 7000, "WMAX": 25000}

    # Redshifts within the default z_range [0.9, 1.7]
    Z_IN_RANGE = np.array([1.0, 1.2, 1.5])

    def _run(self, mock_z_array, z_range=None):
        """Helper: run get_common_grid with mocked I/O and return the result."""
        mock_tbl = {"TARGET_REDSHIFT": mock_z_array}
        kwargs = {} if z_range is None else {"z_range": z_range}
        with patch("process_spectra.fits.getheader", return_value=self.MOCK_HEADER), \
             patch("process_spectra.Table.read", return_value=mock_tbl), \
             patch("builtins.open", mock_open()), \
             patch("process_spectra.json.dump"):
            return get_common_grid(self.REF_TRIPLET, "fake_cat.fits", **kwargs)

    def test_returns_dict_with_required_keys(self):
        result = self._run(self.Z_IN_RANGE)
        for key in ("common_min", "common_max", "n_galaxies_considered", "z_target"):
            assert key in result

    def test_common_min_is_less_than_common_max(self):
        result = self._run(self.Z_IN_RANGE)
        assert result["common_min"] < result["common_max"]

    def test_json_dump_called_once(self):
        mock_tbl = {"TARGET_REDSHIFT": self.Z_IN_RANGE}
        with patch("process_spectra.fits.getheader", return_value=self.MOCK_HEADER), \
             patch("process_spectra.Table.read", return_value=mock_tbl), \
             patch("builtins.open", mock_open()), \
             patch("process_spectra.json.dump") as mock_dump:
            get_common_grid(self.REF_TRIPLET, "fake_cat.fits")

        mock_dump.assert_called_once()

    def test_writes_to_correct_output_path(self):
        mock_tbl = {"TARGET_REDSHIFT": self.Z_IN_RANGE}
        with patch("process_spectra.fits.getheader", return_value=self.MOCK_HEADER), \
             patch("process_spectra.Table.read", return_value=mock_tbl), \
             patch("builtins.open", mock_open()) as mo, \
             patch("process_spectra.json.dump"):
            get_common_grid(self.REF_TRIPLET, "fake_cat.fits")

        mo.assert_called_with("./common_region.json", "w")

    def test_out_of_range_redshifts_excluded_from_count(self):
        # 0.5 and 2.0 are outside [0.9, 1.7]; only 1.0, 1.2, 1.5 should be counted
        z_mixed = np.array([0.5, 1.0, 1.2, 1.5, 2.0])
        result = self._run(z_mixed, z_range=[0.9, 1.7])
        assert result["n_galaxies_considered"] == 3

    def test_n_galaxies_matches_in_range_count(self):
        result = self._run(self.Z_IN_RANGE)
        assert result["n_galaxies_considered"] == len(self.Z_IN_RANGE)
