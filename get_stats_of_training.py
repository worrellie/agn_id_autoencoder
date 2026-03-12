
import h5py
import numpy as np

def compute_and_save_stats(h5_path):
    with h5py.File(h5_path, 'a') as hf: # 'a' for append/edit mode
        # 1. Pull the training data
        # Using a slice [:] loads it into RAM; if too big, use a loop
        train_flux = hf['train']['normalized_flux'][:]
        
        # 2. Create a mask to ignore gaps (0.0)
        mask = (train_flux != 0)
        
        # 3. Calculate Global Stats
        # We only want the stats of the actual physical data
        valid_data = train_flux[mask]
        global_mean = np.mean(valid_data)
        global_std = np.std(valid_data)
        
        # 4. Save as root attributes
        hf.attrs['train_mean'] = global_mean
        hf.attrs['train_std'] = global_std
        
        print(f"✅ Saved to {h5_path}:")
        print(f"   Mean: {global_mean:.6f}")
        print(f"   Std:  {global_std:.6f}")

# Run it once
compute_and_save_stats("all_spectra_normalized.h5")