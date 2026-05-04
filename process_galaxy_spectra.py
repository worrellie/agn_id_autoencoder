import funcs_process_gals as funcs
import os
from specutils.manipulation import FluxConservingResampler
from joblib import Parallel, delayed
import multiprocessing
import h5py
import argparse


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--folder", "-f", default=r"test_all_spectra_sf_q")

	# setup

	args = parser.parse_args()

	input_dir = args.folder

	t = 1  # 1: noisy, 4: template
	exps = [1, 2, 4, 8]

	# Define output directory
	# normalised = "norm_" if norm else ""
	noise_type = "noisy" if t == 1 else "noiseless"
	output_dir = f"processed_{noise_type}_{input_dir}"

	os.makedirs(output_dir, exist_ok=True)

	h5_filename = rf"{input_dir}.h5"

	# run code

	grid_size = 4.0

	common_vals, valid_triplets = funcs.get_common_grid(input_dir, de_z=0.8)

	resampler = FluxConservingResampler(extrapolation_treatment="truncate")

	if os.environ.get("SLURM_CPUS_PER_TASK") is not None:
		print("running on cluster")
		cpus = os.environ.get("SLURM_CPUS_PER_TASK")
		print(f"Starting parallel processing on {cpus} cores...")
	else:
		print("running on non-cluster")
		cpus = multiprocessing.cpu_count() - 1  # Leave one core for the OS
		print(f"Starting parallel processing on {cpus} cores...")

	results = Parallel(n_jobs=cpus)(
		delayed(funcs.process_single_spec)(
			triplet, common_vals, grid_size, output_dir, resampler
		)
		for triplet in valid_triplets
	)

	# at this point, should have a folder of all the processed spectra
	files, train_files, valid_files, test_files = funcs.sklearn_split_data(
		output_dir, h5_filename
	)
	# files, train_files, valid_files, test_files = funcs.sklearn_split_data(output_dir, h5_filename, test_size = test_size)

	# print(train_files)
	# print(valid_files)
	# print(test_files)

	funcs.save_h5(h5_filename, files, train_files, valid_files, test_files)

	# update h5 with train set stats
	# compute_and_save_stats('test_all_spectra.h5', )

	# check h5
	with h5py.File(h5_filename, "r") as hf:
		print(f"\n📑 Root Attributes:")
		for attr in hf.attrs:
			print(f"  - {attr}: {hf.attrs[attr]}")

		print("\n🌳 File Structure:")
		hf.visititems(funcs.check_h5_structure)
	print("/n---------------------------------------------/n")

	# check_h5_samples(h5_filename, norm = False)
	funcs.check_h5_samples(h5_filename, norm=True)


if __name__ == "__main__":
	main()
