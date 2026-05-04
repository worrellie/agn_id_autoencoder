import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

def check_h5(file): 

	H5_FILE = file

	splits = ['train', 'validation', 'test']
	colors = ['steelblue', 'tomato', 'seagreen']

	snr_data = {}

	with h5py.File(H5_FILE, 'r') as hf:
		for split in splits:
			snrs = hf[split]['SNR'][:]
			snr_data[split] = snrs
			print(f"{split:12s} | n={len(snrs):4d} | mean SNR = {np.mean(snrs):.3f} | median = {np.median(snrs):.3f} | std = {np.std(snrs):.3f}")

	# histogram
	fig, ax = plt.subplots(figsize=(10, 5))

	for (split, snrs), color in zip(snr_data.items(), colors):
		ax.hist(snrs, bins=30, alpha=0.6, label=f'{split} (mean={np.mean(snrs):.2f})', color=color, edgecolor='white')

	ax.set_xlabel('SNR')
	ax.set_ylabel('Count')
	ax.set_title('SNR Distribution by Split')
	ax.legend()
	plt.tight_layout()
	plt.savefig(f'snr_distributions_{H5_FILE}.png', dpi=150)
	print(f"\nSaved: snr_distributions_{H5_FILE}.png")


# files = ['all_spectra_to_process_sf.h5', 'all_spectra_to_process_q.h5']
# for f in files:

#	 check_h5(f)


# check_h5("test_all_spectra_sf_q")

# import h5py
# import numpy as np

# with h5py.File('all_spectra_to_process_sf.h5', 'r') as hf:
#	 print("Keys in train:", list(hf['train'].keys()))
#	 s = hf['train']['normalized_flux'][0]
#	 print("First spectrum:", s[:10])
#	 print("Unique values:", np.unique(s))
#	 print("Mean:", s.mean())

with h5py.File('test_all_spectra_sf_q.h5', 'r') as hf:
	print(f"norm_mean: {hf.attrs['norm_mean']}")
	print(f"norm_std:  {hf.attrs['norm_std']}")
	print(f"raw_mean:  {hf.attrs['raw_mean']}")
	print(f"raw_std:   {hf.attrs['raw_std']}")
	
	# also check a few individual normalized spectra means
	print("\nMean of 5 random normalized_flux spectra (train):")
	random_idxs = random.sample(range(len(hf['train']['normalized_flux'])), 5)
	print(random_idxs)
	for i in random_idxs:
		s = hf['train']['normalized_flux'][i]
		mask = s != 0
		print(f"  spec {i}: full mean={s.mean():.6f}, unmasked mean={s[mask].mean():.6f}")