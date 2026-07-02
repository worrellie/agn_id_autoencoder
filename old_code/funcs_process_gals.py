RND = 42

from joblib import Parallel, delayed
import multiprocessing

import glob
import numpy as np
from astropy.io import fits
import re
import joblib
import random
import sys

from matplotlib import pyplot as plt

from specutils import Spectrum
import astropy.units as u
from specutils.manipulation import FluxConservingResampler

from specutils import SpectralRegion

from sklearn.model_selection import train_test_split

import h5py
import os
import random
from astropy.table import Table

import warnings

# np.set_printoptions(threshold=np.inf)


def get_common_grid(input_dir, exps=[1, 2, 4, 8], de_z=0.8):

	# quickly runs through all fits files to get the common wavelength range
	# of all files once they are de-redshifted.
	# returns: common range as a length 2 list: [min, max]
	# all valid file triplets as list of tuples:[(ri, yj, h, z), ...]

	all_rest_mins = []
	all_rest_maxs = []
	valid_file_triplets = []

	for exp_time in exps:
		chan_path = os.path.join(input_dir, f"*_{exp_time}h_z*_RI.fits")

		ri_files = glob.glob(chan_path)

		if not ri_files:
			print(
				f"  Warning: No files found for chan_path: {os.path.abspath(chan_path)}"
			)
		else:
			print(f"  Found {len(ri_files)} files.")

		for ri_path in ri_files:
			# Match redshift from filename
			match = re.search(r"z(\d+\.\d+)", ri_path)
			if not match:
				continue
			z = float(match.group(1))

			# Check for all channels
			yj_path = ri_path.replace("_RI.fits", "_YJ.fits")
			h_path = ri_path.replace("_RI.fits", "_H.fits")

			if os.path.exists(yj_path) and os.path.exists(h_path):
				try:
					wav_ri = fits.getdata(ri_path, ext=9)
					wav_h = fits.getdata(h_path, ext=9)

					obs_min = wav_ri.min()
					obs_max = wav_h.max()

					# de-redshift to certain redshift (0.8 default)
					all_rest_mins.append((1 + de_z) * obs_min / (1 + z))
					all_rest_maxs.append((1 + de_z) * obs_max / (1 + z))
					valid_file_triplets.append((ri_path, yj_path, h_path, z))
				except Exception as e:
					print(f"Error reading {ri_path}: {e}")

	# common overlap when de-redshifted to 0.8
	common_min = np.max(all_rest_mins)
	common_max = np.min(all_rest_maxs)
	# common wavelength grid at 0.8
	# common_grid = np.arange(np.ceil(common_min), np.floor(common_max), 4.0) * u.AA

	print(f"common range: {common_min:.2f} to {common_max:.2f} Å")
	print(f"num specs to process: {len(valid_file_triplets)}")

	return [common_min, common_max], valid_file_triplets


def get_channel_data(channel_path, t=1):

	# returns raw wavelength, flux and template of given channel
	# print(fits.getheader(channel_path))
	# exit()

	flux = fits.getdata(channel_path, ext=t)
	f_templ = fits.getdata(channel_path, ext=4)
	l = fits.getdata(channel_path, ext=9)

	return flux, l, f_templ


def deredshift_channel(flux, l, z, de_z=0.8):

	# returns de-redshifted wavelength and flux

	flux_z = flux * ((1 + z) / (1 + de_z))
	# flux_z = flux

	l_z = (1 + de_z) * (l / (1 + z))

	# gap = np.arange(, , , 0.5)
	# gap_de_z = (1 + de_z) * (l / (1 + z))

	# print(z)
	# print(flux_z, l_z)

	return flux_z, l_z


def rebin_channel(flux, l, resampler, grid_size=4.0):

	# resamples a channel to a common grid and creates a mask (unsure why i did this...)

	assert getattr(resampler, "extrapolation_treatment") == "truncate", (
		"Resampler must truncate values outside new grid"
	)

	master_grid = np.arange(0, 30000, grid_size)

	# shrink master grid to channel size for efficiency
	# *note* this also will crop up to 4AA, to make uniform
	mask = (master_grid >= np.min(l)) & (master_grid <= np.max(l))
	channel_grid = master_grid[mask]
	channel_grid = channel_grid * u.AA

	chan = Spectrum(flux * u.Unit("erg cm-2 s-1 AA-1"), l * u.AA)

	rebinned_chan = resampler(chan, channel_grid)

	# print(len(channel_grid))
	# print(len(l), len(rebinned_chan.spectral_axis.value))
	# print(len(flux), len(rebinned_chan.flux.value))
	# print(l, rebinned_chan.spectral_axis.value)
	# print(flux, rebinned_chan.flux.value)
	# print(min(rebinned_chan.spectral_axis), max(rebinned_chan.spectral_axis))
	# print(rebinned_chan.flux.value, rebinned_chan.spectral_axis.value)

	return rebinned_chan.flux.value, rebinned_chan.spectral_axis.value


def merge_channels(channel_pairs, grid_size):
	# combines all 3 channels and MASKS potentially questionable pixels with 0
	#

	full_l = np.arange(0, 30000, grid_size)

	full_flux = np.zeros_like(full_l)

	prev_end = None
	for f, l in channel_pairs:
		start_idx = int(np.round((l[0] - full_l[0]) / grid_size))
		end_idx = start_idx + len(f)

		# check overlap is not more than 1 or 2 pixels
		if prev_end is not None:
			overlap = start_idx - prev_end
			if 100 <= abs(overlap) <= 300:
				# print('instrument gap')
				pass
			elif -2 <= overlap <= 2:
				# print(f"expected gap")
				pass
			else:
				print(f"unexpected gap!! {overlap}")
		prev_end = end_idx

		# if masking edges with 0
		f_masked = f.copy()
		f_masked[0] = 0
		f_masked[-1] = 0

		full_flux[start_idx:end_idx] = f_masked

	return full_flux, full_l


def crop_spectrum(flux, l, common_vals):

	common_min = np.ceil(common_vals[0])
	common_max = np.floor(common_vals[1])

	idxs = np.where((l >= common_min) & (l <= common_max))
	# print(len(idxs[0]))
	# print(idxs)
	cropped_flux = flux[idxs]
	cropped_l = l[idxs]

	# print(len(cropped_l ))
	# print(min(cropped_l))
	# print(max(cropped_l))

	return cropped_flux, cropped_l


def merge_orignal_de_z(flux, l):

	flux = np.asarray(flux)
	l = np.asarray(l)

	sort_idx = np.argsort(l)
	l = l[sort_idx]
	flux = flux[sort_idx]

	# spec = Spectrum(flux * u.Unit('erg cm-2 s-1 AA-1'), l * u.AA)

	return [flux, l]


def plot_spec(flux, l, z, original=None, templ=None):

	fig, ax = plt.subplots(figsize=(12, 6))
	if original is not None:
		plt.plot(original[1], original[0], color="black", linewidth=0.5)
	plt.step(l, flux, color="red", linewidth=0.5)
	if templ is not None:
		plt.plot(templ[1], templ[0], color="blue", linewidth=0.5)
	# mask
	mask = flux == 0
	padded = np.concatenate(([0], mask.astype(int), [0]))
	diff = np.diff(padded)
	starts = np.where(diff == 1)[0]
	ends = np.where(diff == -1)[0]
	for s, e in zip(starts, ends):
		# We use l[s] to l[e-1] to get the wavelength coordinates
		# Note: if e-1 is out of bounds, we use the last lambda value
		stop_idx = min(e, len(l) - 1)
		ax.axvspan(l[s], l[stop_idx], color="grey", alpha=0.8, label="Masked")

	plt.title(f"z_obsv = {z}")
	plt.show()


def calc_SNR(flux, l):

	flux = np.asarray(flux)
	l = np.asarray(l)

	target_mask = (l >= 5100) & (l <= 5800) & (flux != 0)

	target_flux = flux[target_mask]
	# print(len(target_flux))
	if target_flux.size < 2:
		print("could not get snr: continuum region too small")
		return 0.0, 0.0, 0.0

	noise = np.std(target_flux)
	mean_flux = np.mean(target_flux)
	# median_flux = np.median(target_flux)

	if noise == 0:
		print("could not get snr, zero noise")
		return mean_flux, 0.0, 0.0

	snr = mean_flux / noise

	return mean_flux, noise, snr


def save_spec( flux, l, original_z, snr, norm_factors, infile_base, outdir, noise_type="noisy"):

	col1 = fits.Column(name="lambda", format="D", array=l)
	col2 = fits.Column(name="flux", format="D", array=flux)

	hdr = fits.Header()
	hdr["OG_Z"] = original_z
	hdr["SNR"] = snr
	hdr["ORIGINAL"] = infile_base
	hdr["NORM_CON"] = norm_factors['continuum_mean']
	hdr['NORM_MED'] = norm_factors['full_spec_median']

	hdu = fits.BinTableHDU.from_columns([col1, col2], header=hdr)

	out_name = f"{infile_base}_{noise_type}_deZ_rebinned.fits"

	hdu.writeto(os.path.join(outdir, out_name), overwrite=True)

	return


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

						d_z[i] = redshift
						d_snr[i] = snr
						d_ids[i] = os.path.basename(f).encode("utf-8")

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


def save_zarr():

	# not implemented
	# consider this for speeding up ml
	# cannot do parallel stuff with hdf5

	return None


def process_single_spec(triplet, common_vals, grid_size, output_dir, ):

	resampler = FluxConservingResampler(extrapolation_treatment="truncate")

	ri_p, yj_p, h_p, redshift = triplet
	base_name = os.path.basename(ri_p).replace("_RI.fits", "")

	try:
		original_de_z_flux, original_de_z_l = [], []
		original_flux_rest, original_l_rest = [], []
		channel_pairs = []

		for channel in [ri_p, yj_p, h_p]:
			flux, l, _ = get_channel_data(
				channel
			)  # Assuming t=1 is hardcoded or passed

			flux_de_z, l_de_z = deredshift_channel(flux, l, redshift, de_z=0.8)
			flux_rest, l_rest = deredshift_channel(flux, l, redshift, de_z=0.0)

			original_de_z_flux.extend(flux_de_z)
			original_de_z_l.extend(l_de_z)
			original_flux_rest.extend(flux_rest)
			original_l_rest.extend(l_rest)

			f_rebinned, l_rebinned = rebin_channel(
				flux_de_z, l_de_z, resampler, grid_size=grid_size
			)
			channel_pairs.append([f_rebinned, l_rebinned])

		# merge channels and get final spectrum
		spec_flux, spec_l = merge_channels(channel_pairs, grid_size=grid_size)
		final_spec_flux, final_spec_l = crop_spectrum(spec_flux, spec_l, common_vals)
		
		# check for fully masked spectra
		mask = final_spec_flux == 0
		if mask.all():
			print(f"fully masked spec: {base_name}")

		# get normalization factors
		cont_mean, noise, snr = calc_SNR( np.asarray(original_flux_rest), np.asarray(original_l_rest))
		if (noise == 0.0 and snr == 0.0) or (cont_mean == 0.0 or np.isnan(cont_mean) or cont_mean is None):
			print(f"invalid spec {base_name} with ({cont_mean}, {noise}, {snr})")

		full_spec_median = np.median(final_spec_flux[~mask])
		if  (full_spec_median == 0.0 or np.isnan(full_spec_median) or full_spec_median is None):
			print(f"invalid spec {base_name} with {full_spec_median}")
		
		norm_factors = {'continuum_mean' : cont_mean,
						'full_spec_median' : full_spec_median}

		# save spectrum in fits
		save_spec(final_spec_flux, final_spec_l, redshift, snr, norm_factors, base_name, output_dir,)

		return base_name  # Useful for tracking progress

	except Exception as e:
		print(
			f"\nERROR: Worker failed on file: {base_name}", file=sys.stderr, flush=True
		)
		print(f"Error details: {e}")
		# Re-raise the error if you want the whole job to stop,
		# or return None if you want the job to keep going for other files
		raise e
