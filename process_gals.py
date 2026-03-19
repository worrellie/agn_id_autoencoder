import os
import glob
import numpy as np
from astropy.io import fits
import re
import joblib
import random

from matplotlib import pyplot as plt

from specutils import Spectrum
import astropy.units as u
from specutils.manipulation import FluxConservingResampler

from specutils import SpectralRegion

import h5py
import os
import glob
import numpy as np
from astropy.io import fits
from sklearn.model_selection import train_test_split

import os
import random
import matplotlib.pyplot as plt
from astropy.table import Table

np.set_printoptions(threshold=np.inf)


def get_common_grid(input_dir, exps = [1, 2, 4, 8]):

    all_rest_mins = []
    all_rest_maxs = []
    valid_file_triplets = []

    for exp_time in exps:
        pattern = os.path.join(input_dir, f"*_{exp_time}h_RI.fits")
        ri_files = glob.glob(pattern)

        for ri_path in ri_files:
            # Match redshift from filename
            match = re.search(r'z(\d+\.\d+)', ri_path)
            if not match: continue
            z = float(match.group(1))

            # Check for all channels
            yj_path = ri_path.replace('_RI.fits', '_YJ.fits')
            h_path = ri_path.replace('_RI.fits', '_H.fits')
            
            if os.path.exists(yj_path) and os.path.exists(h_path):
                # Speed fix: Use fits.getdata to only grab the wavelength extension (idx 9)
                # This is much faster than fits.open() which maps the whole file
                try:
                    wav_ri = fits.getdata(ri_path, ext=9)
                    wav_h = fits.getdata(h_path, ext=9)
                    
                    obs_min = wav_ri.min()
                    obs_max = wav_h.max()
                    
                    # de-redshift to 0.8
                    all_rest_mins.append(1.8 * obs_min / (1 + z))
                    all_rest_maxs.append(1.8 * obs_max / (1 + z))
                    valid_file_triplets.append((ri_path, yj_path, h_path, z))
                except Exception as e:
                    print(f"Error reading {ri_path}: {e}")


    # common overlap when de-redshifted to 0.8
    common_min = np.max(all_rest_mins)
    common_max = np.min(all_rest_maxs)
    # common wavelength grid at 0.8
    common_grid = np.arange(np.ceil(common_min), np.floor(common_max), 4.0) * u.AA

    print(f"✅ Common grid found: {common_min:.2f} to {common_max:.2f} Å")
    print(f"Total files to process: {len(valid_file_triplets)}")

    return common_grid, valid_file_triplets

def deredshift_and_resample(common_grid, valid_file_triplets, norm, output_dir):

    resampler = FluxConservingResampler()

    i = 0
    # for each channel of each file
    for ri_p, yj_p, h_p, redshift in valid_file_triplets:
        # get file base name (remove channel part)
        base_name = os.path.basename(ri_p).replace('_RI.fits', '')
        
        combined_lambda = []
        combined_flux = []
        bounds = [] # boundaries of channels to get intrument gap position

        for p in [ri_p, yj_p, h_p]:
            with fits.open(p) as hdul:
                flux = hdul[t].data
                wav = hdul[9].data
                combined_lambda.append(wav)
                combined_flux.append(flux)
                bounds.append((wav.min(), wav.max()))

        # get mask from first spec only (so it is uniform)
        # (it should be th same everywhere anyway but por si las moscas)
        if i == 0:
            mask_min = bounds[1][1] + 1
            mask_max = bounds[2][0] - 1
            instrument_gap = np.arange(start = mask_min, stop= mask_max, step = 1)
            instrument_flux = [np.nan] * len(instrument_gap)
        # include gap 'data'
        combined_lambda.append(instrument_gap)
        combined_flux.append(instrument_flux)

        # sort flux according to lambda 
        final_lambda = np.concatenate(combined_lambda) 
        final_flux = np.concatenate(combined_flux)
        sort_idx = np.argsort(final_lambda)
        final_lambda = final_lambda[sort_idx]
        final_flux = final_flux[sort_idx]

        # get mask where instrument gap is (true where masking)
        instrument_mask = np.isnan(final_flux)

        # unit
        final_lambda = final_lambda * u.AA
        final_flux = final_flux * u.Unit('erg cm-2 s-1 AA-1')

        spec_rest = Spectrum(
            spectral_axis=1.8*final_lambda/(1+redshift),
            flux=final_flux,
            mask = instrument_mask,
        )

        # rebin to common grid
        rebinned_spec = resampler(spec_rest, common_grid)

        flux_vals = rebinned_spec.flux.value # Convert to raw numpy array
        valid_pixels = flux_vals[~rebinned_spec.mask]
        masked_pixels = flux_vals[rebinned_spec.mask]

        # if normalizing, normalize (how lol)
        if norm:
            # mean = valid_pixels.mean()
            # std = valid_pixels.std()
            # flux_vals_norm = (flux_vals - mean)/ std
            norm_factor = 1
            flux_vals_norm = flux_vals
            flux_vals = flux_vals_norm

        # Update the plot to show the enforced gaps
        if base_name == valid_file_triplets[0][0].split('/')[-1].replace('_RI.fits', ''):
            plt.figure(figsize=(12, 6))
            plt.step(spec_rest.spectral_axis, spec_rest.flux, 
                    where='mid', color='gray', alpha=0.3, label='Input (Rest)')
            plt.step(common_grid, flux_vals, 
                    where='mid', color='crimson', lw=1.5, label='Rebinned & Masked')
            
            # Highlight where the gaps were enforced
            plt.fill_between(common_grid.value, 0, np.nanmax(flux_vals), 
                            where=rebinned_spec.mask, color='black', alpha=0.1, label='Enforced Gap')
            
            plt.title(f"Verification: {base_name} (z={redshift})")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{base_name}_check.png"))
            plt.show()
            plt.close()

        # --- SAVE TO FITS ---
        # make masked pixels 0 instead 0
        flux_vals = np.nan_to_num(flux_vals, nan=0)

        col1 = fits.Column(name='lambda', format='D', array=common_grid.value)
        col2 = fits.Column(name='flux', format='D', array=flux_vals) # Use the masked flux

        hdr = fits.Header()
        hdr['REDSHIFT'] = redshift
        hdr['ORIGINAL'] = base_name
        if norm:
            hdr['NORMFAC'] = norm_factor # Important to keep for science!
        
        hdu = fits.BinTableHDU.from_columns([col1, col2], header=hdr)
        
        out_name = f"{base_name}_{noise_type}_rebinned.fits"
        
        hdu.writeto(os.path.join(output_dir, out_name), overwrite=True)

        i += 1


def plot_random_fits_tables(folder_path, num_files=5):
    # 1. Filter for .fits files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    
    if not all_files:
        print("No .fits files found in the directory.")
        return

    num_files = min(len(all_files), num_files)
    selected_files = random.sample(all_files, num_files)
    
    # 2. Setup Plotting
    fig, axes = plt.subplots(num_files, 1, figsize=(8, 4 * num_files), sharex=False)
    if num_files == 1: axes = [axes]

    for i, filename in enumerate(selected_files):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 3. Read the FITS table
            # Astropy automatically finds the first extension containing a table
            t = Table.read(file_path)
            
            # Get column names (assuming there are at least 2)
            cols = t.colnames
            # print(cols)
            x_col, y_col = cols[0], cols[1]
            # print(t[x_col], t[y_col])
            
            # 4. Plotting
            axes[i].plot(t[x_col], t[y_col], linestyle='-', alpha=0.7)
            axes[i].set_title(f"File: {filename}")
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_col)
            axes[i].grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            axes[i].set_title(f"Error: {filename}")

    plt.tight_layout()
    plt.show()

def compile_to_h5_split_sklearn(processed_dir, norm, h5_filename):
    # 1. Get the list of all files (matching your new naming convention)
    files = np.array(sorted(glob.glob(os.path.join(processed_dir, "*_rebinned.fits"))))
    
    if len(files) == 0:
        print(f"No files found in {processed_dir}! Check your path and naming.")
        return

    # # 2. Split the file list (80% train, 10% val, 10% test)
    # train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
    # valid_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # for testing:
    train_files, temp_files = train_test_split(files, test_size=0.5, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)


    # 3. Peek at dimensions from the first file
    with fits.open(files[0]) as hdul:
        n_pixels = len(hdul[1].data['lambda'])
        wavelength_grid = hdul[1].data['lambda']
    
    file_splits = {
        'train': train_files,
        'validation': valid_files,
        'test': test_files
    }

    # 4. Create the H5 File
    with h5py.File(h5_filename, 'w') as hf:
        # Save common wavelength grid as a root attribute
        hf.attrs['wavelengths'] = wavelength_grid
        # print(wavelength_grid)
        
        for split_name, split_list in file_splits.items():
            n_samples = len(split_list)
            print(f"📦 Writing {split_name} group ({n_samples} samples)...")
            
            group = hf.create_group(split_name)
            
            # Datasets using float64 (f8) for high precision as requested
            # We now create TWO flux datasets: raw and normalized
            if norm:
                d_flux_norm = group.create_dataset('normalized_flux', (n_samples, n_pixels), 
                                                dtype='f8', compression='gzip', chunks=True)
                d_norm_fac = group.create_dataset('norm_factor', (n_samples,), dtype='f8')

            d_flux_raw  = group.create_dataset('raw_flux', (n_samples, n_pixels), 
                                              dtype='f8', compression='gzip', chunks=True)
            d_z = group.create_dataset('redshift', (n_samples,), dtype='f8')
            d_ids = group.create_dataset('obj_id', (n_samples,), dtype='S100')

            # 5. Fill datasets incrementally
            for i, f in enumerate(split_list):
                try:
                    with fits.open(f) as hdul:
                        # Since your processing code saved the normalized flux as the main 'flux' column:
                        # If you didn't save raw separately in the FITS, we can back-calculate 
                        # or just store the normalized one if that's all that's in the FITS.
                        # I'll assume you saved the normalized version to the FITS.
                        
                        current_flux = hdul[1].data['flux'].astype(np.float64)
                        redshift = hdul[0].header.get('REDSHIFT', 0.0)
                        if norm:
                            norm_factor = hdul[0].header.get('NORMFAC', 1.0)
                            d_flux_norm[i] = current_flux
                            d_norm_fac[i] = norm_factor
                        d_z[i] = redshift

                        d_ids[i] = os.path.basename(f).encode('utf-8')
                        
                        # If you want to store the "raw" physical flux, we multiply back
                        if norm:
                            d_flux_raw[i] = current_flux * norm_factor
                        else: 
                            d_flux_raw[i] = current_flux 
                
                except Exception as e:
                    print(f"Skipping {f} due to error: {e}")
                
                if (i + 1) % 100 == 0:
                    print(f"  {split_name} progress: {i+1}/{n_samples}")

    print(f"\n🏁 Successfully compiled {h5_filename}")

def compute_and_save_stats(h5_path, norm):
    with h5py.File(h5_path, 'a') as hf: # 'a' for append/edit mode
        # 1. Pull the training data
        # Using a slice [:] loads it into RAM; if too big, use a loop
        if norm:
            train_flux = hf['train']['normalized_flux'][:]
        else:
            train_flux = hf['train']['raw_flux'][:]
        
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
        print(f"   Mean: {global_mean}")
        print(f"   Std:  {global_std}")

def check_h5_samples(h5_path, norm):
    """
    Checks random samples from the H5 file.
    flux_type: 'normalized_flux' or 'raw_flux'
    """
    if norm:
        flux_type='normalized_flux'
    else:
        flux_type='raw_flux'

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
            if norm:
                norm_fac = hf[split]['norm_factor'][rand_idx]
            
            # Print to console for manual zero-check in the gaps
            print(f"--- {split.upper()} (Index {rand_idx}) ---")
            if norm:
                print(f"ID: {obj_id}, Norm Factor: {norm_fac:.4e}")
            else:
                print(f"ID: {obj_id}")
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

def print_h5_structure(name, obj):
    """Recursive function to print group and dataset info."""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

##########################################################################################################

input_dir = '/home/worrellie/Documents/phd/autoencoder/test_data_for_processing'
t = 1  # 1: noisy, 4: template
norm = False # whether to normalize spectra
exps = [1, 2, 4, 8]

# Define output directory
normalised = "norm_" if norm else ""
noise_type = "noisy" if t == 1 else "noiseless"
output_dir = f'test_processed_z09_z08_{normalised}{noise_type}'

os.makedirs(output_dir, exist_ok=True)

h5_filename = 'test_all_spectra.h5'

#####################################

common_grid, valid_file_triplets = get_common_grid(input_dir)

deredshift_and_resample(common_grid, valid_file_triplets, norm, output_dir)

plot_random_fits_tables(output_dir)

compile_to_h5_split_sklearn(output_dir, norm, h5_filename)

compute_and_save_stats(h5_filename, norm)

with h5py.File(h5_filename, 'r') as hf:
    print(f"\n📑 Root Attributes:")
    for attr in hf.attrs:
        print(f"  - {attr}: {hf.attrs[attr]}")
    
    print("\n🌳 File Structure:")
    hf.visititems(print_h5_structure)
print("/n---------------------------------------------/n")

check_h5_samples(h5_filename, norm)