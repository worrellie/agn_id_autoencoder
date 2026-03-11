import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

# Useful for debugging gaps to see if they are exactly 0.0
np.set_printoptions(threshold=np.inf)

def check_h5_samples(h5_path, flux_type='normalized_flux'):
    """
    Checks random samples from the H5 file.
    flux_type: 'normalized_flux' or 'raw_flux'
    """
    with h5py.File(h5_path, 'r') as hf:
        # 1. Access the wavelength grid from the root attributes
        if 'wavelengths' not in hf.attrs:
            print("Error: 'wavelengths' attribute not found!")
            return
        wave = hf.attrs['wavelengths']
        
        # 2. Setup the plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # Match the group names from your compilation script
        splits = ['train', 'validation', 'test']
        
        for i, split in enumerate(splits):
            if split not in hf:
                print(f"Warning: Group '{split}' not found in file.")
                continue
                
            # Access the chosen flux dataset
            dset = hf[split][flux_type]
            n_samples = dset.shape[0]
            
            # 3. Pick a random index
            rand_idx = random.randint(0, n_samples - 1)
            
            # 4. Load the data
            flux = dset[rand_idx]
            z = hf[split]['redshift'][rand_idx]
            obj_id = hf[split]['obj_id'][rand_idx].decode('utf-8')
            norm_fac = hf[split]['norm_factor'][rand_idx]
            
            # Print to console for manual zero-check in the gaps
            print(f"--- {split.upper()} (Index {rand_idx}) ---")
            print(f"ID: {obj_id}, Norm Factor: {norm_fac:.4e}")
            # print(flux) # Uncomment if you want the full array in console
            
            # 5. Plotting
            axes[i].step(wave, flux, where='mid', color='midnightblue', lw=0.8)
            axes[i].set_title(f"Split: {split.upper()} | ID: {obj_id} | z: {z:.4f} | Type: {flux_type}")
            axes[i].set_ylabel("Normalized Flux" if 'norm' in flux_type else "Raw Flux")
            axes[i].grid(alpha=0.3)
            
            # Highlight zeros (gaps) for visual confirmation
            # Only plot where flux is exactly 0
            gaps = np.where(flux == 0)[0]
            if len(gaps) > 0:
                axes[i].plot(wave[gaps], flux[gaps], 'r|', markersize=2, alpha=0.3, label='Zero-Gap')

        axes[2].set_xlabel(f"Rest-frame Wavelength ($\AA$)")
        plt.tight_layout()
        plt.show()

#############################################################

def print_h5_structure(name, obj):
    """Recursive function to print group and dataset info."""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

###############################################################

h5_path = "test_all_spectra_normalized.h5"

with h5py.File(h5_path, 'r') as hf:
    print(f"\n📑 Root Attributes:")
    for attr in hf.attrs:
        print(f"  - {attr}: {hf.attrs[attr]}")
    
    print("\n🌳 File Structure:")
    hf.visititems(print_h5_structure)
print("/n---------------------------------------------/n")
check_h5_samples(h5_path, flux_type='normalized_flux')