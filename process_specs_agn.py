import os
import glob
import numpy as np
from astropy.io import fits

# Define the paths to your three folders
# Update these to the actual paths on your machine
base_path = '/home/worrellie/Documents/phd/autoencoder/agn'
folders = {
    'RI': os.path.join(base_path, '1h_RI'),
    'YJ': os.path.join(base_path, '1h_YJ'),
    'H':  os.path.join(base_path, '1h_H')
}
output_dir = os.path.join(base_path, 'merged_spectra_agn')
os.makedirs(output_dir, exist_ok=True)

# 1. Get the list of files from the RI folder to use as a template
# Using RI as the 'master' list
pattern = os.path.join(folders['RI'], "*_1h_RI.fits")
ri_files = glob.glob(pattern)

for ri_path in ri_files:
    filename = os.path.basename(ri_path)
    # The common prefix for matching (e.g., AGN_temp_z0.5_ebv0.2_L300045.0_emline1.5_1h)
    common_prefix = filename.replace('_RI.fits', '')
    
    # 2. Construct paths for the other folders
    parts = {
        'RI': ri_path,
        'YJ': os.path.join(folders['YJ'], f"{common_prefix}_YJ.fits"),
        'H':  os.path.join(folders['H'],  f"{common_prefix}_H.fits")
    }

    # Verify all parts exist before processing
    if not all(os.path.exists(p) for p in parts.values()):
        print(f"Skipping {common_prefix}: Missing parts in other folders.")
        continue

    combined_lambda = []
    combined_flux = []

    # 3. Read and merge
    for arm in ['RI', 'YJ', 'H']:
        with fits.open(parts[arm]) as hdul:
            # Check if data is in Extension 0 or 1 (usually 1 for tables)
            # idx = 1 if len(hdul) > 1 else 0
            flux = hdul[1].data
            l = hdul[9].data
            combined_lambda.append(l)
            combined_flux.append(flux)

    final_lambda = np.concatenate(combined_lambda)
    final_flux = np.concatenate(combined_flux)
    
    # Sort by wavelength to ensure a continuous spectrum
    sort_idx = np.argsort(final_lambda)
    final_lambda = final_lambda[sort_idx]
    final_flux = final_flux[sort_idx]

    # 4. Save
    col1 = fits.Column(name='lambda', format='D', array=final_lambda)
    col2 = fits.Column(name='flux', format='D', array=final_flux)
    hdu = fits.BinTableHDU.from_columns([col1, col2])
    
    output_name = f"{common_prefix}_merged.fits"
    hdu.writeto(os.path.join(output_dir, output_name), overwrite=True)
    print(f"Successfully merged: {output_name}")