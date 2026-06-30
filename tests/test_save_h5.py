"""
Test suite for save_h5.py

All tests are fully self-contained: no real FITS or HDF5 files are read from disk.
Every file I/O call is replaced with a unittest.mock patch or a fixture-supplied
numpy array. Generated .h5 files use pytest's tmp_path fixture.

Organised into one class per public function, mirroring the style of
tests/test_process_spectra.py.
"""

import os
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from save_h5 import check_h5_samples
from save_h5 import discover_param_keys
from save_h5 import save_h5 as save_h5_fn
from save_h5 import sklearn_split_data

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

N_PIXELS = 10
LAMBDA_ARR = np.linspace(6500.0, 18000.0, N_PIXELS)
TRAIN_FLUX = np.arange(1.0, 11.0)   # [1, 2, …, 10] — non-trivial, deterministic
NORM_CON = 2.0
NORM_MED = 1.5
OG_Z = 1.2
SNR_VAL = 15.0
PROCESSED_DIR = "fake_processed_spectra"

# Pre-computed expected Welford statistics for 2 identical train files,
# each contributing TRAIN_FLUX (all pixels non-zero, no NaN).
# The function uses population variance: Var = (Σx²/n) − mean²,
# which is identical to numpy's default ddof=0 std.
_norm_cont = TRAIN_FLUX / NORM_CON
_norm_med = TRAIN_FLUX / NORM_MED
_log_vals = np.log1p(TRAIN_FLUX / NORM_CON)   # sign=+1, unmasked

EXPECTED_RAW_MEAN = np.mean(TRAIN_FLUX)          # 5.5
EXPECTED_RAW_STD = np.std(TRAIN_FLUX)            # √8.25 ≈ 2.872
EXPECTED_NORM_CONT_MEAN = np.mean(_norm_cont)
EXPECTED_NORM_CONT_STD = np.std(_norm_cont)
EXPECTED_NORM_MED_MEAN = np.mean(_norm_med)
EXPECTED_NORM_MED_STD = np.std(_norm_med)
EXPECTED_LOG_MEAN = np.mean(_log_vals)
EXPECTED_LOG_STD = np.std(_log_vals)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_filenames(n, tags=None):
    """Return n fake *_rebinned.fits paths, cycling through tags if given."""
    if tags is None:
        return [
            os.path.join(PROCESSED_DIR, f"cosmos_{i:04d}_rebinned.fits")
            for i in range(n)
        ]
    return [
        os.path.join(
            PROCESSED_DIR,
            f"cosmos_{i:04d}_{tags[i % len(tags)]}_rebinned.fits",
        )
        for i in range(n)
    ]


def _make_mock_hdul(
    flux_arr=None,
    lambda_arr=None,
    norm_con=NORM_CON,
    norm_med=NORM_MED,
    og_z=OG_Z,
    snr=SNR_VAL,
    extra_header_vals=None,
):
    """
    Build a MagicMock that satisfies the astropy HDUList context-manager
    protocol and subscript pattern used in save_h5.py:

        with fits.open(path) as hdul:
            hdul[1].data["lambda"]      → lambda_arr
            hdul[1].data["flux"]        → flux_arr (real ndarray, .astype works)
            hdul[1].header.get(key)     → header_vals[key]
            hdul[1].header[key]         → header_vals[key]   (subscript access)
            hdul[1].header.cards        → [] (empty — discover_param_keys returns [])
    """
    if flux_arr is None:
        flux_arr = TRAIN_FLUX.copy()
    if lambda_arr is None:
        lambda_arr = LAMBDA_ARR.copy()

    header_vals = {
        "NORM_CON": norm_con,
        "NORM_MED": norm_med,
        "OG_Z": og_z,
        "SNR": snr,
    }
    if extra_header_vals:
        header_vals.update(extra_header_vals)

    mock_data = MagicMock()
    mock_data.__getitem__ = MagicMock(
        side_effect=lambda key: lambda_arr if key == "lambda" else flux_arr
    )

    mock_header = MagicMock()
    mock_header.get = MagicMock(side_effect=lambda key, *a: header_vals.get(key))
    mock_header.__getitem__ = MagicMock(side_effect=lambda key: header_vals[key])
    mock_header.cards = []

    mock_hdu1 = MagicMock()
    mock_hdu1.data = mock_data
    mock_hdu1.header = mock_header

    mock_hdul = MagicMock()
    mock_hdul.__enter__ = MagicMock(return_value=mock_hdul)
    mock_hdul.__exit__ = MagicMock(return_value=False)
    mock_hdul.__getitem__ = MagicMock(return_value=mock_hdu1)

    return mock_hdul


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def h5_path(tmp_path):
    return str(tmp_path / "test_output.h5")


@pytest.fixture
def minimal_h5(tmp_path):
    """Fully-populated but tiny .h5 file for check_h5_samples tests."""
    path = tmp_path / "minimal.h5"
    n_rows = 3
    norm_cont_flux = TRAIN_FLUX / NORM_CON
    norm_med_flux = TRAIN_FLUX / NORM_MED
    log_flux = np.log1p(norm_cont_flux)

    with h5py.File(str(path), "w") as hf:
        hf.attrs["wavelengths"] = LAMBDA_ARR
        for split in ["train", "validation", "test"]:
            grp = hf.create_group(split)
            grp.create_dataset("raw_flux", data=np.tile(TRAIN_FLUX, (n_rows, 1)))
            grp.create_dataset(
                "normalized_flux_cont", data=np.tile(norm_cont_flux, (n_rows, 1))
            )
            grp.create_dataset(
                "normalized_flux_med", data=np.tile(norm_med_flux, (n_rows, 1))
            )
            grp.create_dataset(
                "log_scale_flux", data=np.tile(log_flux, (n_rows, 1))
            )
            grp.create_dataset("redshift", data=np.full(n_rows, OG_Z))
            grp.create_dataset("SNR", data=np.full(n_rows, SNR_VAL))
            ids = [f"fake_obj_{k}".encode("utf-8") for k in range(n_rows)]
            grp.create_dataset("obj_id", data=np.array(ids))

    return str(path)


# ---------------------------------------------------------------------------
# 1. sklearn_split_data
# ---------------------------------------------------------------------------


class TestSklearnSplitData:
    """Patches save_h5.glob.glob; never touches the real filesystem."""

    def test_no_files_returns_none(self):
        with patch("save_h5.glob.glob", return_value=[]):
            result = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert result is None

    def test_returns_four_tuple(self):
        fake = _make_fake_filenames(8)
        with patch("save_h5.glob.glob", return_value=fake):
            result = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert isinstance(result, tuple) and len(result) == 4

    def test_files_array_contains_all_paths(self):
        fake = _make_fake_filenames(8)
        with patch("save_h5.glob.glob", return_value=fake):
            files, *_ = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert set(files) == set(fake)

    # ── Small dataset (≤ 10 files) ──────────────────────────────────────────

    def test_small_dataset_counts_sum_to_total(self):
        fake = _make_fake_filenames(8)
        with patch("save_h5.glob.glob", return_value=fake):
            files, train, valid, test = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert len(train) + len(valid) + len(test) == len(files)

    def test_small_dataset_train_is_majority(self):
        fake = _make_fake_filenames(8)
        with patch("save_h5.glob.glob", return_value=fake):
            _, train, valid, test = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert len(train) > len(valid)
        assert len(train) > len(test)

    def test_small_dataset_all_subsets_nonempty(self):
        # 8 files × 0.2 = 2 temp → 1 valid, 1 test; train gets the rest
        fake = _make_fake_filenames(8)
        with patch("save_h5.glob.glob", return_value=fake):
            _, train, valid, test = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert len(train) > 0
        assert len(valid) > 0
        assert len(test) > 0

    # ── Large dataset (> 10 files, per-group split) ─────────────────────────

    def test_large_dataset_counts_sum_to_total(self):
        # 32 files: 8 per tag × 4 tags — triggers the per-group branch
        fake = _make_fake_filenames(32, tags=["1h", "2h", "4h", "8h"])
        with patch("save_h5.glob.glob", return_value=fake):
            files, train, valid, test = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert len(train) + len(valid) + len(test) == len(files)

    def test_large_dataset_train_is_majority(self):
        fake = _make_fake_filenames(32, tags=["1h", "2h", "4h", "8h"])
        with patch("save_h5.glob.glob", return_value=fake):
            _, train, valid, test = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert len(train) > len(valid)
        assert len(train) > len(test)

    def test_large_dataset_all_four_tags_represented_in_train(self):
        # Every tag should contribute files to the training set
        fake = _make_fake_filenames(32, tags=["1h", "2h", "4h", "8h"])
        with patch("save_h5.glob.glob", return_value=fake):
            _, train, *_ = sklearn_split_data(PROCESSED_DIR, "out.h5")
        for tag in ["1h", "2h", "4h", "8h"]:
            assert any(tag in f for f in train), f"tag '{tag}' missing from train"

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_missing_tag_group_skipped_without_error(self):
        # "4h" and "8h" are absent; function must still return a valid split
        fake = _make_fake_filenames(20, tags=["1h", "2h"])
        with patch("save_h5.glob.glob", return_value=fake):
            result = sklearn_split_data(PROCESSED_DIR, "out.h5")
        assert result is not None
        files, train, valid, test = result
        assert len(train) + len(valid) + len(test) == len(files)

    def test_five_file_edge_case_raises_sklearn_error(self):
        # With 5 files, test_size=0.2 → 4 train + 1 temp.
        # sklearn cannot split 1 temp sample 50/50 (train set would be empty)
        # → documents a known limitation for very small datasets.
        fake = _make_fake_filenames(5)
        with patch("save_h5.glob.glob", return_value=fake):
            with pytest.raises(ValueError, match="train set will be empty"):
                sklearn_split_data(PROCESSED_DIR, "out.h5")


# ---------------------------------------------------------------------------
# 2. save_h5 — split: 2 train, 1 valid, 1 test
# ---------------------------------------------------------------------------

_TRAIN_FILES = ["f0.fits", "f1.fits"]
_VALID_FILES = ["f2.fits"]
_TEST_FILES = ["f3.fits"]
_ALL_FILES = _TRAIN_FILES + _VALID_FILES + _TEST_FILES


class TestSaveH5:
    """
    save_h5_fn is called once per test via the autouse fixture.
    Each test gets its own tmp_path → h5_path, and reads results from the
    written HDF5 file.
    """

    @pytest.fixture(autouse=True)
    def _write_h5(self, h5_path):
        hdul = _make_mock_hdul()
        with patch("save_h5.fits.open", return_value=hdul):
            save_h5_fn(
                "ref.fits", h5_path, _ALL_FILES, _TRAIN_FILES, _VALID_FILES, _TEST_FILES
            )
        self.h5_path = h5_path

    # ── File / group structure ───────────────────────────────────────────────

    def test_h5_file_is_created(self):
        assert os.path.exists(self.h5_path)

    def test_h5_has_train_group(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert "train" in hf

    def test_h5_has_validation_group(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert "validation" in hf

    def test_h5_has_test_group(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert "test" in hf

    def test_wavelengths_root_attr_stored(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(hf.attrs["wavelengths"], LAMBDA_ARR)

    # ── Dataset shapes ───────────────────────────────────────────────────────

    def test_train_raw_flux_shape(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["raw_flux"].shape == (2, N_PIXELS)

    def test_valid_raw_flux_shape(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["validation"]["raw_flux"].shape == (1, N_PIXELS)

    def test_test_raw_flux_shape(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["test"]["raw_flux"].shape == (1, N_PIXELS)

    # ── Metadata stored correctly ────────────────────────────────────────────

    def test_obj_ids_stored(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["obj_id"][0] == b"f0.fits"
            assert hf["train"]["obj_id"][1] == b"f1.fits"

    def test_redshift_stored(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(hf["train"]["redshift"][:], OG_Z)

    def test_snr_stored(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(hf["train"]["SNR"][:], SNR_VAL)

    # ── Flux transformations ─────────────────────────────────────────────────

    def test_raw_flux_values_correct(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(hf["train"]["raw_flux"][0], TRAIN_FLUX)

    def test_norm_cont_values_correct(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(
                hf["train"]["normalized_flux_cont"][0], TRAIN_FLUX / NORM_CON
            )

    def test_norm_med_values_correct(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(
                hf["train"]["normalized_flux_med"][0], TRAIN_FLUX / NORM_MED
            )

    def test_log_scale_values_correct(self):
        # All TRAIN_FLUX values are positive and non-zero:
        # log_scale = sign(norm_cont) * log1p(|norm_cont|) * unmasked = log1p(norm_cont)
        expected = np.log1p(TRAIN_FLUX / NORM_CON)
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(hf["train"]["log_scale_flux"][0], expected)

    # ── Welford's algorithm (train-set global statistics) ───────────────────
    # 2 train files × 10 pixels each; population variance = Σx²/n − mean²
    # This equals np.std(TRAIN_FLUX, ddof=0) because both files are identical.

    def test_welford_raw_mean(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["raw_mean"] == pytest.approx(EXPECTED_RAW_MEAN, abs=1e-10)

    def test_welford_raw_std(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["raw_std"] == pytest.approx(EXPECTED_RAW_STD, abs=1e-10)

    def test_welford_norm_cont_mean(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["norm_mean_cont"] == pytest.approx(
                EXPECTED_NORM_CONT_MEAN, abs=1e-10
            )

    def test_welford_norm_cont_std(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["norm_std_cont"] == pytest.approx(
                EXPECTED_NORM_CONT_STD, abs=1e-10
            )

    def test_welford_norm_med_mean(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["norm_mean_med"] == pytest.approx(
                EXPECTED_NORM_MED_MEAN, abs=1e-10
            )

    def test_welford_norm_med_std(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["norm_std_med"] == pytest.approx(
                EXPECTED_NORM_MED_STD, abs=1e-10
            )

    def test_welford_log_mean(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["norm_mean_log"] == pytest.approx(
                EXPECTED_LOG_MEAN, abs=1e-10
            )

    def test_welford_log_std(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["norm_std_log"] == pytest.approx(
                EXPECTED_LOG_STD, abs=1e-10
            )

    def test_all_welford_attrs_present_in_h5(self):
        expected_keys = {
            "raw_mean", "raw_std",
            "norm_mean_cont", "norm_std_cont",
            "norm_mean_med", "norm_std_med",
            "norm_mean_log", "norm_std_log",
        }
        with h5py.File(self.h5_path, "r") as hf:
            assert expected_keys.issubset(set(hf.attrs.keys()))


# ---------------------------------------------------------------------------
# 3. save_h5 — invalid normalisation-factor guard-rails
# ---------------------------------------------------------------------------


class TestSaveH5NormGuards:
    """
    Tests for the NORM_CON / NORM_MED validation branch.
    Each test creates a fresh file via tmp_path.
    """

    def _run(self, h5_path, norm_con=NORM_CON, norm_med=NORM_MED):
        """Run save_h5_fn with one file per split and the given norm factors.
        Each split must be non-empty: h5py requires chunk size ≤ dataset rows,
        and the default chunk is (1, N_PIXELS), so (0, N_PIXELS) datasets fail.
        """
        hdul = _make_mock_hdul(norm_con=norm_con, norm_med=norm_med)
        train = ["f0.fits"]
        valid = ["f1.fits"]
        test = ["f2.fits"]
        files = train + valid + test
        with patch("save_h5.fits.open", return_value=hdul):
            save_h5_fn("ref.fits", h5_path, files, train, valid, test)

    def test_norm_con_zero_emits_warning(self, h5_path):
        with pytest.warns(UserWarning, match="NORM_CON"):
            self._run(h5_path, norm_con=0)

    def test_norm_con_none_emits_warning(self, h5_path):
        with pytest.warns(UserWarning, match="NORM_CON"):
            self._run(h5_path, norm_con=None)

    def test_norm_con_nan_emits_warning(self, h5_path):
        with pytest.warns(UserWarning, match="NORM_CON"):
            self._run(h5_path, norm_con=float("nan"))

    def test_norm_med_zero_emits_warning(self, h5_path):
        with pytest.warns(UserWarning, match="NORM_MED"):
            self._run(h5_path, norm_med=0)

    def test_norm_con_zero_defaults_to_1_so_norm_cont_equals_raw(self, h5_path):
        # When NORM_CON is invalid → defaults to 1.0 → norm_cont = raw / 1.0 = raw
        with pytest.warns(UserWarning):
            self._run(h5_path, norm_con=0)
        with h5py.File(h5_path, "r") as hf:
            assert np.allclose(hf["train"]["normalized_flux_cont"][0], TRAIN_FLUX)

    def test_norm_med_none_defaults_to_1_so_norm_med_equals_raw(self, h5_path):
        with pytest.warns(UserWarning):
            self._run(h5_path, norm_med=None)
        with h5py.File(h5_path, "r") as hf:
            assert np.allclose(hf["train"]["normalized_flux_med"][0], TRAIN_FLUX)


# ---------------------------------------------------------------------------
# 4. check_h5_samples
# ---------------------------------------------------------------------------


class TestCheckH5Samples:
    """
    Uses the minimal_h5 fixture (a real but tiny .h5 file).
    Patches save_h5.plt so no PDFs are generated.
    """

    def _call(self, path, norm=False):
        """Call check_h5_samples under a full plt mock; return the mock."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        with patch("save_h5.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            check_h5_samples(path, norm=norm)
        return mock_plt

    def test_savefig_called_with_correct_filename(self, minimal_h5):
        mock_plt = self._call(minimal_h5)
        mock_plt.savefig.assert_called_once_with("eg_samples.pdf")

    def test_subplots_called_with_3_rows_1_col(self, minimal_h5):
        mock_plt = self._call(minimal_h5)
        args, _ = mock_plt.subplots.call_args
        assert args[0] == 3
        assert args[1] == 1

    def test_tight_layout_called(self, minimal_h5):
        mock_plt = self._call(minimal_h5)
        mock_plt.tight_layout.assert_called_once()

    def test_missing_wavelengths_attr_returns_early(self, tmp_path):
        # Without the root 'wavelengths' attribute the function should return
        # immediately, never reaching plt.subplots.
        path = str(tmp_path / "no_wave.h5")
        with h5py.File(path, "w") as hf:
            hf.create_group("train")   # no wavelengths attr

        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        with patch("save_h5.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            check_h5_samples(path, norm=False)

        mock_plt.subplots.assert_not_called()

    def test_norm_true_still_calls_savefig(self, minimal_h5):
        # norm=True uses the normalized_flux_cont dataset path; output is the same
        mock_plt = self._call(minimal_h5, norm=True)
        mock_plt.savefig.assert_called_once_with("eg_samples.pdf")

    def test_missing_split_group_handled_gracefully(self, tmp_path):
        # If one of the expected splits is absent, the function should
        # continue rather than raise a KeyError.
        path = str(tmp_path / "partial.h5")
        n_rows = 2
        with h5py.File(path, "w") as hf:
            hf.attrs["wavelengths"] = LAMBDA_ARR
            # Only "train" present; "validation" and "test" are absent
            grp = hf.create_group("train")
            grp.create_dataset("raw_flux", data=np.tile(TRAIN_FLUX, (n_rows, 1)))
            grp.create_dataset(
                "normalized_flux_cont",
                data=np.tile(TRAIN_FLUX / NORM_CON, (n_rows, 1)),
            )
            grp.create_dataset(
                "normalized_flux_med",
                data=np.tile(TRAIN_FLUX / NORM_MED, (n_rows, 1)),
            )
            grp.create_dataset(
                "log_scale_flux",
                data=np.tile(np.log1p(TRAIN_FLUX / NORM_CON), (n_rows, 1)),
            )
            grp.create_dataset("redshift", data=np.full(n_rows, OG_Z))
            grp.create_dataset("SNR", data=np.full(n_rows, SNR_VAL))
            grp.create_dataset(
                "obj_id",
                data=np.array([b"fake_0", b"fake_1"]),
            )

        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        with patch("save_h5.plt") as mock_plt:
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            try:
                check_h5_samples(path, norm=False)
            except Exception as exc:
                pytest.fail(
                    f"check_h5_samples raised unexpectedly with missing splits: {exc}"
                )


# ---------------------------------------------------------------------------
# 5. discover_param_keys
# ---------------------------------------------------------------------------


def _make_hdul_with_cards(card_keywords):
    """Return a mock HDUList whose header.cards yields the given keywords."""
    mock_cards = []
    for kw in card_keywords:
        card = MagicMock()
        card.keyword = kw
        mock_cards.append(card)

    mock_header = MagicMock()
    mock_header.cards = mock_cards

    mock_hdu1 = MagicMock()
    mock_hdu1.header = mock_header

    mock_hdul = MagicMock()
    mock_hdul.__enter__ = MagicMock(return_value=mock_hdul)
    mock_hdul.__exit__ = MagicMock(return_value=False)
    mock_hdul.__getitem__ = MagicMock(return_value=mock_hdu1)
    return mock_hdul


class TestDiscoverParamKeys:
    """discover_param_keys filters FITS boilerplate and system keys from a header."""

    def _run(self, card_keywords):
        hdul = _make_hdul_with_cards(card_keywords)
        with patch("save_h5.fits.open", return_value=hdul):
            return discover_param_keys("fake.fits")

    def test_empty_header_returns_empty_list(self):
        assert self._run([]) == []

    def test_returns_list_type(self):
        assert isinstance(self._run(["_REDSHI"]), list)

    def test_system_keys_excluded(self):
        from save_h5 import SYSTEM_KEYS
        result = self._run(list(SYSTEM_KEYS) + ["_REDSHI"])
        for key in SYSTEM_KEYS:
            assert key not in result

    def test_fits_boilerplate_excluded(self):
        boilerplate = ["BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT",
                       "TFIELDS", "SIMPLE", "EXTEND", "COMMENT", "HISTORY"]
        result = self._run(boilerplate + ["_REDSHI"])
        for key in boilerplate:
            assert key not in result

    def test_empty_string_keyword_excluded(self):
        result = self._run(["", "_REDSHI"])
        assert "" not in result
        assert "_REDSHI" in result

    def test_tform_prefix_excluded(self):
        result = self._run(["TFORM1", "TFORM22", "_REDSHI"])
        assert "TFORM1" not in result
        assert "TFORM22" not in result

    def test_ttype_prefix_excluded(self):
        result = self._run(["TTYPE1", "_REDSHI"])
        assert "TTYPE1" not in result

    def test_tunit_prefix_excluded(self):
        result = self._run(["TUNIT3", "_REDSHI"])
        assert "TUNIT3" not in result

    def test_catalogue_keys_returned(self):
        result = self._run(["OG_Z", "_REDSHI", "HMAG", "MAG_I"])
        assert "_REDSHI" in result
        assert "HMAG" in result
        assert "MAG_I" in result

    def test_order_preserved(self):
        keywords = ["HMAG", "_REDSHI", "MAG_I"]
        result = self._run(keywords)
        assert result == keywords


# ---------------------------------------------------------------------------
# 6. save_h5 — per-spectrum catalogue parameter datasets
# ---------------------------------------------------------------------------


class TestSaveH5ParamDatasets:
    """
    When discover_param_keys returns non-empty keys, save_h5 must create one
    dataset per key in every split group and populate it from the FITS header.
    """

    _PARAM_KEY = "_REDSHI"
    _PARAM_VAL = 1.1

    @pytest.fixture(autouse=True)
    def _write_h5(self, h5_path):
        hdul = _make_mock_hdul(extra_header_vals={self._PARAM_KEY: self._PARAM_VAL})
        with patch("save_h5.fits.open", return_value=hdul), \
             patch("save_h5.discover_param_keys", return_value=[self._PARAM_KEY]):
            save_h5_fn(
                "ref.fits", h5_path, _ALL_FILES, _TRAIN_FILES, _VALID_FILES, _TEST_FILES
            )
        self.h5_path = h5_path

    def test_param_dataset_created_in_train(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert self._PARAM_KEY in hf["train"]

    def test_param_dataset_created_in_validation(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert self._PARAM_KEY in hf["validation"]

    def test_param_dataset_created_in_test(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert self._PARAM_KEY in hf["test"]

    def test_param_dataset_shape_matches_split_size(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"][self._PARAM_KEY].shape == (len(_TRAIN_FILES),)
            assert hf["validation"][self._PARAM_KEY].shape == (len(_VALID_FILES),)
            assert hf["test"][self._PARAM_KEY].shape == (len(_TEST_FILES),)

    def test_param_values_stored_from_header(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert np.allclose(hf["train"][self._PARAM_KEY][:], self._PARAM_VAL)


# ---------------------------------------------------------------------------
# 7. save_h5 — skip-and-trim behaviour (row counter + dataset resize)
# ---------------------------------------------------------------------------

# 3-file train split so we can have: succeed, fail, succeed
_SKIP_TRAIN = ["fa.fits", "fb.fits", "fc.fits"]
_SKIP_VALID = ["fd.fits"]
_SKIP_TEST  = ["fe.fits"]
_SKIP_ALL   = _SKIP_TRAIN + _SKIP_VALID + _SKIP_TEST


def _open_with_one_bad_file(path):
    """fits.open side_effect: raises OSError for fb.fits, succeeds for everything else."""
    if os.path.basename(path) == "fb.fits":
        raise OSError("corrupt FITS")
    return _make_mock_hdul()


class TestSaveH5SkipHandling:
    """
    Verify the row-counter / resize / skipped-attr pattern:
    - datasets are over-allocated then trimmed after any FITS errors
    - skipped filenames are recorded in group.attrs["skipped"]
    - successful files before and after the error are written correctly
    - Welford stats exclude the erroring file
    """

    @pytest.fixture(autouse=True)
    def _write_h5(self, h5_path):
        with patch("save_h5.fits.open", side_effect=_open_with_one_bad_file), \
             patch("save_h5.discover_param_keys", return_value=[]):
            save_h5_fn(
                "ref.fits", h5_path, _SKIP_ALL, _SKIP_TRAIN, _SKIP_VALID, _SKIP_TEST
            )
        self.h5_path = h5_path

    # ── Dataset trimming ─────────────────────────────────────────────────────

    def test_train_raw_flux_trimmed_to_successful_rows(self):
        # 3 train files, 1 errored → shape (2, N_PIXELS)
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["raw_flux"].shape == (2, N_PIXELS)

    def test_train_redshift_trimmed_to_successful_rows(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["redshift"].shape == (2,)

    def test_train_snr_trimmed_to_successful_rows(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["SNR"].shape == (2,)

    def test_train_obj_id_trimmed_to_successful_rows(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["obj_id"].shape == (2,)

    # ── Skipped attribute ────────────────────────────────────────────────────

    def test_skipped_attr_stored_on_erroring_group(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert "skipped" in hf["train"].attrs

    def test_skipped_attr_contains_erroring_filename(self):
        with h5py.File(self.h5_path, "r") as hf:
            skipped = list(hf["train"].attrs["skipped"])
            assert b"fb.fits" in skipped

    def test_no_skipped_attr_when_all_succeed(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert "skipped" not in hf["validation"].attrs
            assert "skipped" not in hf["test"].attrs

    # ── Correct data for successful files ────────────────────────────────────

    def test_first_successful_file_obj_id(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["obj_id"][0] == b"fa.fits"

    def test_file_after_skip_also_written(self):
        # fc.fits (index 2) comes after the erroring fb.fits; it must land at row 1
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["train"]["obj_id"][1] == b"fc.fits"

    def test_successful_splits_are_full_size(self):
        with h5py.File(self.h5_path, "r") as hf:
            assert hf["validation"]["raw_flux"].shape == (1, N_PIXELS)
            assert hf["test"]["raw_flux"].shape      == (1, N_PIXELS)

    # ── Welford stats exclude the erroring file ───────────────────────────────

    def test_welford_stats_only_count_successful_files(self):
        # fa and fc both contribute TRAIN_FLUX; fb never opened → same expected
        # values as normal 2-file case (both files have identical TRAIN_FLUX)
        with h5py.File(self.h5_path, "r") as hf:
            assert hf.attrs["raw_mean"] == pytest.approx(EXPECTED_RAW_MEAN, abs=1e-10)
            assert hf.attrs["raw_std"]  == pytest.approx(EXPECTED_RAW_STD,  abs=1e-10)
