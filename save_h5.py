
RND = 42

# from joblib import Parallel, delayed
# import multiprocessing

import glob
import numpy as np
from astropy.io import fits
# import re
# import joblib
# import random
# import sys

from matplotlib import pyplot as plt

# from specutils import Spectrum
# import astropy.units as u
# from specutils.manipulation import FluxConservingResampler

# from specutils import SpectralRegion

from sklearn.model_selection import train_test_split

import h5py
import os
import random
# from astropy.table import Table

import warnings


# helper for getting fits hear keys/names
SYSTEM_KEYS = {"OG_Z", "SNR", "ORIGINAL", "NORM_CON", "NORM_MED"}
SKIP = {"XTENSION","BITPIX","NAXIS","NAXIS1","NAXIS2","PCOUNT","GCOUNT",
        "TFIELDS","SIMPLE","EXTEND","COMMENT","HISTORY",""}

def discover_param_keys(sample_fits):
    """Read one processed FITS header, return the catalogue-parameter keywords."""
    with fits.open(sample_fits) as hdul:
        hdr = hdul[1].header
    keys = []
    for card in hdr.cards:
        kw = card.keyword
        if kw in SYSTEM_KEYS or kw in SKIP or kw.startswith(("TFORM","TTYPE","TUNIT")):
            continue
        keys.append(kw)
    return keys



def sklearn_split_data(processed_dir, h5_filename, test_size=0.2, norm=False):

    files = np.array(sorted(glob.glob(os.path.join(processed_dir, "*_rebinned.fits"))))

    if len(files) == 0:
        print(f"No files found in {processed_dir}! Check your path and naming.")
        return

    groups = {
        tag: [f for f in files if tag in os.path.basename(f)]
        for tag in ["1h", "2h", "4h", "8h"]
    }

    all_train = []
    all_valid = []
    all_test = []

    if len(files) > 10:
        for tag, group in groups.items():
            print(f"{tag} has: {len(group)} sources")
            if not group:
                print(f"no files found of {tag} exposure time")
                continue  # leave loop iteration and go to next
            train_files, temp_files = train_test_split(
                group, test_size=test_size, random_state=RND
            )
            valid_files, test_files = train_test_split(
                temp_files, test_size=0.5, random_state=RND
            )
            all_train.extend(train_files)
            all_valid.extend(valid_files)
            all_test.extend(test_files)
    else:
        train_files, temp_files = train_test_split(
            files, test_size=test_size, random_state=RND
        )
        valid_files, test_files = train_test_split(
            temp_files, test_size=0.5, random_state=RND
        )
        all_train.extend(train_files)
        all_valid.extend(valid_files)
        all_test.extend(test_files)


    return files, all_train, all_valid, all_test


def save_h5(reference_fits, h5_filename, files, train_files, valid_files, test_files):

    with fits.open(files[0]) as hdul:
        n_pixels = len(hdul[1].data["lambda"])
        wavelength_grid = hdul[1].data["lambda"]

    file_splits = {"train": train_files, "validation": valid_files, "test": test_files}
    train_stats = {}

    # schema is identical across splits -> discover keywords ONCE
    param_keys = discover_param_keys(reference_fits)

    with h5py.File(h5_filename, "w") as hf:
        hf.attrs["wavelengths"] = wavelength_grid

        for split_name, split_list in file_splits.items():
            n_samples = len(split_list)
            print(f"📦 Writing {split_name} group ({n_samples} samples)...")

            group = hf.create_group(split_name)

            # maxshape=(None, ...) makes axis 0 resizable so we can trim skips later
            d_flux_log = group.create_dataset(
                "log_scale_flux", (n_samples, n_pixels), dtype="f8",
                maxshape=(None, n_pixels), compression="gzip", chunks=(1, n_pixels),
            )
            d_flux_norm_cont = group.create_dataset(
                "normalized_flux_cont", (n_samples, n_pixels), dtype="f8",
                maxshape=(None, n_pixels), compression="gzip", chunks=(1, n_pixels),
            )
            d_flux_norm_med = group.create_dataset(
                "normalized_flux_med", (n_samples, n_pixels), dtype="f8",
                maxshape=(None, n_pixels), compression="gzip", chunks=(1, n_pixels),
            )
            d_flux_raw = group.create_dataset(
                "raw_flux", (n_samples, n_pixels), dtype="f8",
                maxshape=(None, n_pixels), compression="gzip", chunks=(1, n_pixels),
            )

            d_z   = group.create_dataset("redshift", (n_samples,), dtype="f8",  maxshape=(None,))
            d_snr = group.create_dataset("SNR",      (n_samples,), dtype="f8",  maxshape=(None,))
            d_ids = group.create_dataset("obj_id",   (n_samples,), dtype="S100", maxshape=(None,))

            param_dsets = {
                kw: group.create_dataset(kw, (n_samples,), dtype="f8", maxshape=(None,))
                for kw in param_keys
            }

            # training-set running stats (population) — your existing sum-based form
            total_pixels = 0
            sum_raw = sum_sq_raw = 0.0
            sum_norm_cont = sum_sq_norm_cont = 0.0
            sum_norm_med = sum_sq_norm_med = 0.0
            sum_norm_log = sum_sq_norm_log = 0.0

            row = 0           # write position: advances ONLY on success
            skipped = []      # audit trail of files that errored

            for f in split_list:
                try:
                    with fits.open(f) as hdul:
                        hdr = hdul[1].header
                        raw_flux = hdul[1].data["flux"].astype(np.float64)
                        unmasked = (raw_flux != 0)

                        norm_factor_continuum = hdr.get("NORM_CON")
                        norm_factor_median = hdr.get("NORM_MED")

                        if (norm_factor_continuum is None or norm_factor_continuum == 0
                                or np.isnan(norm_factor_continuum)):
                            warnings.warn(f"Invalid NORM_CON ({norm_factor_continuum}) in "
                                          f"{os.path.basename(f)}. Defaulting to 1.0.")
                            norm_factor_continuum = 1.0
                        if (norm_factor_median is None or norm_factor_median == 0
                                or np.isnan(norm_factor_median)):
                            warnings.warn(f"Invalid NORM_MED ({norm_factor_median}) in "
                                          f"{os.path.basename(f)}. Defaulting to 1.0.")
                            norm_factor_median = 1.0

                        norm_flux_cont = raw_flux / norm_factor_continuum
                        norm_flux_med  = raw_flux / norm_factor_median
                        log_scale_flux = np.sign(norm_flux_cont) * np.log1p(np.abs(norm_flux_cont))
                        log_scale_flux = log_scale_flux * unmasked

                        # --- writes (use `row`, not the loop position) ---
                        d_z[row]   = float(hdr["OG_Z"])
                        d_snr[row] = float(hdr["SNR"])
                        d_ids[row] = os.path.basename(f).encode("utf-8")
                        for param_name, dset in param_dsets.items():
                            val = hdr.get(param_name, np.nan)
                            dset[row] = np.nan if val is None else val

                        d_flux_raw[row]       = raw_flux
                        d_flux_norm_cont[row] = norm_flux_cont
                        d_flux_norm_med[row]  = norm_flux_med
                        d_flux_log[row]       = log_scale_flux

                        # --- stats LAST: only counted once the row is committed ---
                        if split_name == "train":
                            mask = (
                                (raw_flux != 0)
                                & (~np.isnan(raw_flux))
                                & (~np.isnan(norm_flux_cont))
                                & (~np.isnan(norm_flux_med))
                                & (~np.isnan(log_scale_flux))
                            )
                            valid_raw       = raw_flux[mask]
                            valid_norm_cont = norm_flux_cont[mask]
                            valid_norm_med  = norm_flux_med[mask]
                            valid_log       = log_scale_flux[mask]

                            total_pixels      += valid_raw.size
                            sum_raw           += np.sum(valid_raw)
                            sum_sq_raw        += np.sum(valid_raw**2)
                            sum_norm_cont     += np.sum(valid_norm_cont)
                            sum_sq_norm_cont  += np.sum(valid_norm_cont**2)
                            sum_norm_med      += np.sum(valid_norm_med)
                            sum_sq_norm_med   += np.sum(valid_norm_med**2)
                            sum_norm_log      += np.sum(valid_log)
                            sum_sq_norm_log   += np.sum(valid_log**2)

                    row += 1   # reached only if the whole try-body succeeded

                except Exception as e:
                    print(f"Skipping {f} due to error: {e}")
                    skipped.append(os.path.basename(f))
                    # row NOT advanced -> no hole

            # trim the over-allocated tail left by any skips
            if row < n_samples:
                print(f"{split_name}: {n_samples - row} skipped; resizing {n_samples} -> {row}")
                for dset in [d_flux_log, d_flux_norm_cont, d_flux_norm_med, d_flux_raw,
                             d_z, d_snr, d_ids, *param_dsets.values()]:
                    dset.resize(row, axis=0)

            if skipped:
                group.attrs["skipped"] = np.array(skipped, dtype="S")

            # ---- final training stats (unchanged maths) ----
            if split_name == "train":
                final_mean_raw = sum_raw / total_pixels
                final_std_raw = np.sqrt(max(0, sum_sq_raw / total_pixels - final_mean_raw**2))
                final_mean_norm_cont = sum_norm_cont / total_pixels
                final_std_norm_cont = np.sqrt(max(0, sum_sq_norm_cont / total_pixels - final_mean_norm_cont**2))
                final_mean_norm_med = sum_norm_med / total_pixels
                final_std_norm_med = np.sqrt(max(0, sum_sq_norm_med / total_pixels - final_mean_norm_med**2))
                final_mean_norm_log = sum_norm_log / total_pixels
                final_std_norm_log = np.sqrt(max(0, sum_sq_norm_log / total_pixels - final_mean_norm_log**2))

                train_stats.update({
                    "raw_mean": final_mean_raw, "raw_std": final_std_raw,
                    "norm_mean_cont": final_mean_norm_cont, "norm_std_cont": final_std_norm_cont,
                    "norm_mean_med": final_mean_norm_med, "norm_std_med": final_std_norm_med,
                    "norm_mean_log": final_mean_norm_log, "norm_std_log": final_std_norm_log,
                })

        if "raw_mean" in train_stats:
            for k, v in train_stats.items():
                hf.attrs[k] = v

    print(f"\n🏁 Successfully compiled {h5_filename}")

def save_h5_deprecated(reference_fits, h5_filename, files, train_files, valid_files, test_files):

    # check dims
    with fits.open(files[0]) as hdul:
        n_pixels = len(hdul[1].data["lambda"])
        wavelength_grid = hdul[1].data["lambda"]

    file_splits = {"train": train_files, "validation": valid_files, "test": test_files}

    train_stats = {}

    # 4. Create the H5 File
    with h5py.File(h5_filename, "w") as hf:

        # Save common wavelength grid as a root attribute
        hf.attrs["wavelengths"] = wavelength_grid
        
        # get hdr param names from a reference fits file (random processed spec)
        param_keys = discover_param_keys(reference_fits)

        # for each of train, validation, and test, make datasets for raw,
        # log scale flux, continuum normalized and full spec median normalized
        # as well as snr, redshift and id for each spectrum
        for split_name, split_list in file_splits.items():
            n_samples = len(split_list)
            print(f"📦 Writing {split_name} group ({n_samples} samples)...")

            # make a group for the split
            group = hf.create_group(split_name)

            # make the datasets that will be filled
            d_flux_log = group.create_dataset(
                "log_scale_flux",
                (n_samples, n_pixels),
                dtype="f8",
                compression="gzip",
                chunks=(1, n_pixels),
            )

            d_flux_norm_cont = group.create_dataset(
                "normalized_flux_cont",
                (n_samples, n_pixels),
                dtype="f8",
                compression="gzip",
                chunks=(1, n_pixels),
            )

            d_flux_norm_med = group.create_dataset(
                "normalized_flux_med",
                (n_samples, n_pixels),
                dtype="f8",
                compression="gzip",
                chunks=(1, n_pixels),
            )

            d_flux_raw = group.create_dataset(
                "raw_flux",
                (n_samples, n_pixels),
                dtype="f8",
                compression="gzip",
                chunks=(1, n_pixels),
            )

            d_z = group.create_dataset("redshift", (n_samples,), dtype="f8")
            d_snr = group.create_dataset("SNR", (n_samples,), dtype="f8")
            d_ids = group.create_dataset("obj_id", (n_samples,), dtype="S100")


            # dictionary of the datasets needed for the galaxy params
            param_dsets = {
                kw: group.create_dataset(kw, (n_samples,), dtype="f8")
                for kw in param_keys
            }

            # Welford's algorithm/ running sums for calculation of global means
            # (only done for training datset to avoid data leakage)
            # this is important for data efficiency if have a bazillion files
            total_pixels = 0
            sum_raw = 0.0
            sum_sq_raw = 0.0
            sum_norm_cont = 0.0
            sum_sq_norm_cont = 0.0
            sum_norm_med = 0.0
            sum_sq_norm_med = 0.0
            sum_norm_log = 0.0
            sum_sq_norm_log = 0.0

            # 5. Fill datasets incrementally
            for i, f in enumerate(split_list):
                try:
                    with fits.open(f) as hdul:

                        hdr = hdul[1].header

                        raw_flux = hdul[1].data["flux"].astype(np.float64)
                        unmasked = (raw_flux != 0)

                        norm_factor_continuum = hdul[1].header.get("NORM_CON")
                        norm_factor_median = hdul[1].header.get("NORM_MED")

                        if (norm_factor_continuum is None or norm_factor_continuum == 0 or np.isnan(norm_factor_continuum) ):
                            warnings.warn( f"Invalid NORM_CON ({norm_factor_continuum}) in {os.path.basename(f)}. Defaulting to 1.0.")
                            norm_factor_continuum = 1.0

                        if (norm_factor_median is None or norm_factor_median == 0 or np.isnan(norm_factor_median) ):
                            warnings.warn( f"Invalid NORM_MED ({norm_factor_median}) in {os.path.basename(f)}. Defaulting to 1.0.")
                            norm_factor_median = 1.0

                        norm_flux_cont = raw_flux / norm_factor_continuum
                        norm_flux_med = raw_flux / norm_factor_median
                        ##
                        log_scale_flux = np.sign(norm_flux_cont) * np.log1p(np.abs(norm_flux_cont)) # (log1p(x) is ln(1+x))
                        # log_scale_flux = np.sign(norm_flux_med) * np.log10(np.abs(norm_flux_med)+1)
                        log_scale_flux = log_scale_flux * unmasked

                        # get mean and std of training set for run time normalization
                        if split_name == "train":
                            mask = (
                                (raw_flux != 0)
                                & (~np.isnan(raw_flux))
                                & (~np.isnan(norm_flux_cont))
                                & (~np.isnan(norm_flux_med))
                                & (~np.isnan(log_scale_flux))
                            )
                            valid_raw = raw_flux[mask]
                            valid_norm_cont = norm_flux_cont[mask]
                            valid_norm_med = norm_flux_med[mask]
                            valid_log = log_scale_flux[mask]

                            # for Welford's algorithm
                            total_pixels += valid_raw.size

                            sum_raw += np.sum(valid_raw)
                            sum_sq_raw += np.sum(valid_raw**2)

                            sum_norm_cont += np.sum(valid_norm_cont)
                            sum_sq_norm_cont += np.sum(valid_norm_cont**2)

                            sum_norm_med += np.sum(valid_norm_med)
                            sum_sq_norm_med += np.sum(valid_norm_med**2)

                            sum_norm_log += np.sum(valid_log)
                            sum_sq_norm_log += np.sum(valid_log**2)
                        
                        # redshift = hdul[1].header.get("OG_Z")
                        # snr = hdul[1].header.get("SNR")

                        # d_z[i] = redshift
                        # d_snr[i] = snr
                        d_z[i] = float(hdr["OG_Z"])
                        d_snr[i] = float(hdr["SNR"])
                        d_ids[i] = os.path.basename(f).encode("utf-8")

                        for param_name, dset in param_dsets.items():
                            val = hdr.get(param_name, np.nan)
                            dset[i] =np.nan if val is None else val

                        d_flux_raw[i] = raw_flux
                        d_flux_norm_cont[i] = norm_flux_cont
                        d_flux_norm_med[i] = norm_flux_med
                        d_flux_log[i] = log_scale_flux

                        

                except Exception as e:
                    print(f"Skipping {f} due to error: {e}")

            # final calculations for mean and std of training dataset
            if split_name == "train":
                final_mean_raw = sum_raw / total_pixels
                # Variance = (Sum_of_Squares / n) - (Mean^2)
                final_variance_raw = (sum_sq_raw / total_pixels) - (final_mean_raw**2)
                final_std_raw = np.sqrt(
                    max(0, final_variance_raw)
                )  # max(0,...) prevents tiny negative numbers due to precision

                final_mean_norm_cont = sum_norm_cont / total_pixels
                # Variance = (Sum_of_Squares / n) - (Mean^2)
                final_variance_norm_cont = (sum_sq_norm_cont / total_pixels) - (final_mean_norm_cont**2 )
                final_std_norm_cont = np.sqrt(
                    max(0, final_variance_norm_cont)
                )  # max(0,...) prevents tiny negative numbers due to precision

                final_mean_norm_med = sum_norm_med / total_pixels
                # Variance = (Sum_of_Squares / n) - (Mean^2)
                final_variance_norm_med = (sum_sq_norm_med / total_pixels) - (final_mean_norm_med**2)
                final_std_norm_med = np.sqrt(
                    max(0, final_variance_norm_med)
                )  # max(0,...) prevents tiny negative numbers due to precision

                final_mean_norm_log = sum_norm_log / total_pixels
                # Variance = (Sum_of_Squares / n) - (Mean^2)
                final_variance_norm_log = (sum_sq_norm_log / total_pixels) - (final_mean_norm_log**2)
                final_std_norm_log = np.sqrt(
                    max(0, final_variance_norm_log)
                )  # max(0,...) prevents tiny negative numbers due to precision

                train_stats["raw_mean"] = final_mean_raw
                train_stats["raw_std"] = final_std_raw

                train_stats["norm_mean_cont"] = final_mean_norm_cont
                train_stats["norm_std_cont"] = final_std_norm_cont

                train_stats["norm_mean_med"] = final_mean_norm_med
                train_stats["norm_std_med"] = final_std_norm_med

                train_stats["norm_mean_log"] = final_mean_norm_log
                train_stats["norm_std_log"] = final_std_norm_log

        if "raw_mean" in train_stats:
            hf.attrs["raw_mean"] = train_stats["raw_mean"]
            hf.attrs["raw_std"] = train_stats["raw_std"]

            hf.attrs["norm_mean_cont"] = train_stats["norm_mean_cont"]
            hf.attrs["norm_std_cont"] = train_stats["norm_std_cont"]

            hf.attrs["norm_mean_med"] = train_stats["norm_mean_med"]
            hf.attrs["norm_std_med"] = train_stats["norm_std_med"]

            hf.attrs["norm_mean_log"] = train_stats["norm_mean_log"]
            hf.attrs["norm_std_log"] = train_stats["norm_std_log"]
            

    print(f"\n🏁 Successfully compiled {h5_filename}")

def check_h5_samples(h5_path, norm):
    """
    Checks random samples from the H5 file.
    """

    with h5py.File(h5_path, "r") as hf:
        # 1. Access the wavelength grid from the root attributes
        if "wavelengths" not in hf.attrs:
            print("Error: 'wavelengths' attribute not found!")
            return
        wave = hf.attrs["wavelengths"]

        # 2. Setup the plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Match the group names from your compilation script
        splits = ["train", "validation", "test"]

        # loop for writing
        for i, split in enumerate(splits):
            if split not in hf:
                print(f"Warning: Group '{split}' not found in file.")
                continue

            # Access the chosen flux dataset
            dset_raw = hf[split]["raw_flux"]
            dset_norm_cont = hf[split]["normalized_flux_cont"]
            dset_norm_med = hf[split]["normalized_flux_med"]
            dset_log = hf[split]["log_scale_flux"]
            n_samples = dset_raw.shape[0]

            # 3. Pick a random index
            rand_idx = random.randint(0, n_samples - 1)

            # 4. Load the data
            flux = dset_raw[rand_idx]
            norm_flux = dset_norm_cont[rand_idx]
            z = hf[split]["redshift"][rand_idx]
            obj_id = hf[split]["obj_id"][rand_idx].decode("utf-8")

            # Print to console for manual zero-check in the gaps
            print(f"--- {split.upper()} (Index {rand_idx}) ---")
            print(f"ID: {obj_id}")
            # print(flux) # Uncomment if you want the full array in console

            # 5. Plotting
            if norm:
                axes[i].step(wave, norm_flux, where="mid", color="green", lw=0.8)
                axes[i].set_title(
                    f"Split: {split.upper()} | ID: {obj_id} | z: {z:.4f} (normalized spec)"
                )
            else:
                axes[i].step(wave, flux, where="mid", color="midnightblue", lw=0.8)
                axes[i].set_title(
                    f"Split: {split.upper()} | ID: {obj_id} | z: {z:.4f} "
                )
            axes[i].set_ylabel("Flux")
            axes[i].grid(alpha=0.3)

            # Highlight zeros (gaps) for visual confirmation
            # Only plot where flux is exactly 0
            gaps = np.where(flux == 0)[0]
            if len(gaps) > 0:
                axes[i].plot(
                    wave[gaps],
                    flux[gaps],
                    "r|",
                    markersize=2,
                    alpha=0.3,
                    label="Zero-Gap",
                )

        axes[2].set_xlabel(r"Wavelength ($\AA$)")
        plt.tight_layout()
        # plt.show()
        plt.savefig("eg_samples.pdf")

def check_h5_structure(name, obj):
    """Recursive function to print group and dataset info."""
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

def main():



    output_dir = "processed_spectra"
    all_processed_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir,f))]
    reference_fits = random.choice(all_processed_files) if all_processed_files else None
    h5_filename = "all_spectra.h5"

    files, train_files, valid_files, test_files = sklearn_split_data(
            output_dir, h5_filename
        )
    # files, train_files, valid_files, test_files = funcs.sklearn_split_data(output_dir, h5_filename, test_size = test_size)

    # print(train_files)
    # print(valid_files)
    # print(test_files)

    save_h5(reference_fits, h5_filename, files, train_files, valid_files, test_files)

    # check h5
    with h5py.File(h5_filename, "r") as hf:
        print(f"\n📑 Root Attributes:")
        for attr in hf.attrs:
            print(f"  - {attr}: {hf.attrs[attr]}")

        print("\n🌳 File Structure:")
        hf.visititems(check_h5_structure)
    print("\n---------------------------------------------\n")

    # check_h5_samples(h5_filename, norm = False)
    check_h5_samples(h5_filename, norm=True)

if __name__ == "__main__":
    main()