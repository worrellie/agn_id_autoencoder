import os
import glob
import numpy as np
from astropy.io import fits

# Directory containing your files
input_dir = '/home/worrellie/Documents/phd/autoencoder/Datasets/z08_v3-002'
output_dir = 'merged_z08_v3-002'
os.makedirs(output_dir, exist_ok=True)

# 1. Identify all unique '1h' exposure IDs
# We look for the RI files as a reference point for each unique object
pattern = os.path.join(input_dir, "*_1h_RI.fits")
ri_files = glob.glob(pattern)

for ri_path in ri_files:
    # Extract the base name (e.g., mambo_97000060017188_z0.9035)
    base_name = os.path.basename(ri_path).replace('_RI.fits', '')
    
    # Define the paths for the 3 parts
    parts = {
        'RI': ri_path,
        'YJ': ri_path.replace('_RI.fits', '_YJ.fits'),
        'H':  ri_path.replace('_RI.fits', '_H.fits')
    }
    
    # Check if all 3 parts actually exist
    if not all(os.path.exists(p) for p in parts.values()):
        print(f"Skipping {base_name}: Missing one or more arms.")
        continue

    combined_lambda = []
    combined_flux = []

    # 2. Extract data from each part
    # Order matters: RI (Blue) -> YJ (Green) -> H (Red)
    for arm in ['RI', 'YJ', 'H']:
        with fits.open(parts[arm]) as hdul:
            flux = hdul[1].data
            l = hdul[9].data
            # Assuming columns are named 'lambda' and 'flux'
            # Adjust these strings if your FITS headers use different names
            combined_lambda.append(l)
            combined_flux.append(flux)


    # 3. Concatenate and sort
    # Even if files are read in order, it's safer to sort by wavelength
    final_lambda = np.concatenate(combined_lambda)
    final_flux = np.concatenate(combined_flux)
    
    sort_idx = np.argsort(final_lambda)
    final_lambda = final_lambda[sort_idx]
    final_flux = final_flux[sort_idx]

    # 4. Save to a new FITS file
    col1 = fits.Column(name='lambda', format='D', array=final_lambda)
    col2 = fits.Column(name='flux', format='D', array=final_flux)
    hdu = fits.BinTableHDU.from_columns([col1, col2])
    
    output_filename = f"{base_name}_merged.fits"
    hdu.writeto(os.path.join(output_dir, output_filename), overwrite=True)
    print(f"Created: {output_filename}")