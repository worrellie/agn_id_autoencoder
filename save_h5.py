
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


def save_h5(h5_filename, files, train_files, valid_files, test_files):

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

		# for each of train, validation, and test, make datasets for raw,
		# log scale flux, continuum normalized and full spec median normalized
		# as well as snr, redshift and id for each spectrum
		for split_name, split_list in file_splits.items():
			n_samples = len(split_list)
			print(f"📦 Writing {split_name} group ({n_samples} samples)...")

			group = hf.create_group(split_name)

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
			d_hmag = group.create_dataset("HMAG", (n_samples,), dtype="f8")
			d_alpha = group.create_dataset("ALPHA", (n_samples,), dtype="f8")
			d_beta = group.create_dataset("BETA", (n_samples,), dtype="f8")
			d_tau = group.create_dataset("TAU", (n_samples,), dtype="f8")
			d_massfo = group.create_dataset("MASSFO", (n_samples,), dtype="f8")
			d_metall = group.create_dataset("METALL", (n_samples,), dtype="f8")
			d_dust_av = group.create_dataset("DUST_AV", (n_samples,), dtype="f8")
			d_stellar = group.create_dataset("STELLAR", (n_samples,), dtype="f8")
			d_formed_ = group.create_dataset("FORMED_", (n_samples,), dtype="f8")
			d_sfr = group.create_dataset("SFR", (n_samples,), dtype="f8")
			d_ssfr = group.create_dataset("SSFR", (n_samples,), dtype="f8")
			d_nsfr = group.create_dataset("NSFR", (n_samples,), dtype="f8")
			d_mass_we = group.create_dataset("MASS_WE", (n_samples,), dtype="f8")
			d_tform = group.create_dataset("TFORM", (n_samples,), dtype="f8")
			d_tquench = group.create_dataset("TQUENCH", (n_samples,), dtype="f8")
			d_uv = group.create_dataset("UV_COLO", (n_samples,), dtype="f8")
			d_vj = group.create_dataset("VJ_COLO", (n_samples,), dtype="f8")
			d_u = group.create_dataset("U", (n_samples,), dtype="f8")
			d_v = group.create_dataset("V", (n_samples,), dtype="f8")
			d_j = group.create_dataset("J", (n_samples,), dtype="f8")

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
						
						redshift = hdul[1].header.get("OG_Z")
						snr = hdul[1].header.get("SNR")
						hmag = hdul[1].header.get("_HMAG")
						alpha = hdul[1].header.get("_ALPHA")
						beta = hdul[1].header.get("_BETA")
						tau = hdul[1].header.get("_TAU")
						massfo = hdul[1].header.get("_MASSFO")
						metall = hdul[1].header.get("_METALL")
						dust_av = hdul[1].header.get("DUST_AV")
						stellar = hdul[1].header.get("STELLAR")
						formed_ = hdul[1].header.get("FORMED_")
						sfr = hdul[1].header.get("SFR")
						ssfr = hdul[1].header.get("SSFR")
						nsfr = hdul[1].header.get("NSFR")
						mass_we = hdul[1].header.get("MASS_WE")
						tform = hdul[1].header.get("TFORM")
						tquench = hdul[1].header.get("TQUENCH")
						uv = hdul[1].header.get("UV_COLO")
						vj = hdul[1].header.get("VJ_COLO")
						u = hdul[1].header.get("U")
						v = hdul[1].header.get("V")
						j = hdul[1].header.get("J")

						d_z[i] = redshift
						d_snr[i] = snr
						d_ids[i] = os.path.basename(f).encode("utf-8")
						d_hmag[i] = hmag
						d_alpha[i] = alpha
						d_beta[i] = beta
						d_tau[i] = tau
						d_massfo[i] = massfo
						d_metall[i] = metall
						d_dust_av[i] = dust_av
						d_stellar[i] = stellar
						d_formed_[i] = formed_
						d_sfr[i] = sfr
						d_ssfr[i] = ssfr
						d_nsfr[i] = nsfr
						d_mass_we[i] = mass_we
						d_tform[i] = tform
						d_tquench[i] = tquench
						d_uv[i] = uv
						d_vj[i] = vj
						d_u[i] = u
						d_v[i] = v
						d_j[i] = j

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
	h5_filename = "all_spectra.h5"

	files, train_files, valid_files, test_files = sklearn_split_data(
			output_dir, h5_filename
		)
		# files, train_files, valid_files, test_files = funcs.sklearn_split_data(output_dir, h5_filename, test_size = test_size)

		# print(train_files)
		# print(valid_files)
		# print(test_files)

	save_h5(h5_filename, files, train_files, valid_files, test_files)

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