import h5py
import numpy as np
import matplotlib.pyplot as plt

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

#     check_h5(f)


check_h5("test_all_spectra_sf_q")