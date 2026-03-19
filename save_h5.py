import h5py
import os
import glob
import numpy as np
from astropy.io import fits
from sklearn.model_selection import train_test_split

def compile_to_h5_split_sklearn(processed_dir, h5_filename="test_all_spectra.h5"):
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
        
        for split_name, split_list in file_splits.items():
            n_samples = len(split_list)
            print(f"📦 Writing {split_name} group ({n_samples} samples)...")
            
            group = hf.create_group(split_name)
            
            # Datasets using float64 (f8) for high precision as requested
            # We now create TWO flux datasets: raw and normalized
            d_flux_norm = group.create_dataset('normalized_flux', (n_samples, n_pixels), 
                                              dtype='f8', compression='gzip', chunks=True)
            d_flux_raw  = group.create_dataset('raw_flux', (n_samples, n_pixels), 
                                              dtype='f8', compression='gzip', chunks=True)
            
            d_z = group.create_dataset('redshift', (n_samples,), dtype='f8')
            d_norm_fac = group.create_dataset('norm_factor', (n_samples,), dtype='f8')
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
                        norm_factor = hdul[0].header.get('NORMFAC', 1.0)
                        
                        d_flux_norm[i] = current_flux
                        d_z[i] = redshift
                        d_norm_fac[i] = norm_factor
                        d_ids[i] = os.path.basename(f).encode('utf-8')
                        
                        # If you want to store the "raw" physical flux, we multiply back
                        d_flux_raw[i] = current_flux * norm_factor
                
                except Exception as e:
                    print(f"Skipping {f} due to error: {e}")
                
                if (i + 1) % 100 == 0:
                    print(f"  {split_name} progress: {i+1}/{n_samples}")

    print(f"\n🏁 Successfully compiled {h5_filename}")


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

# Run it
compile_to_h5_split_sklearn("./test_processed_z09_z08_normalized_noisy/")
# Run it once
compute_and_save_stats("test_all_spectra.h5")