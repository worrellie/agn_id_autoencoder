import os
import glob
import numpy as np
from astropy.io import fits
import re
import joblib

from matplotlib import pyplot as plt

from specutils import Spectrum
import astropy.units as u
from specutils.manipulation import FluxConservingResampler

from specutils import SpectralRegion

#####################
input_dir = '/home/worrellie/Documents/phd/autoencoder/z09_and_z08'
t = 1  # 1: noisy, 4: template
lin_scale = False
exps = [1, 2, 4, 8]

# Define output directory
suffix = "linearly_scaled_" if lin_scale else ""
noise_type = "noisy" if t == 1 else "noiseless"
output_dir = f'processed_z09_z08_normalized_{suffix}{noise_type}'
os.makedirs(output_dir, exist_ok=True)

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
                
                all_rest_mins.append(obs_min / (1 + z))
                all_rest_maxs.append(obs_max / (1 + z))
                valid_file_triplets.append((ri_path, yj_path, h_path, z))
            except Exception as e:
                print(f"Error reading {ri_path}: {e}")


# Determine the common overlap (Inner envelope)
common_min = np.max(all_rest_mins)
common_max = np.min(all_rest_maxs)
# Define the common grid (e.g., 1.0 Angstrom spacing)
common_grid = np.arange(np.ceil(common_min), np.floor(common_max), 1.0) * u.AA

print(f"✅ Common grid found: {common_min:.2f} to {common_max:.2f} Å")
print(f"Total files to process: {len(valid_file_triplets)}")


##

resampler = FluxConservingResampler()

for ri_p, yj_p, h_p, redshift in valid_file_triplets:
    base_name = os.path.basename(ri_p).replace('_RI.fits', '')
    
    combined_lambda = []
    combined_flux = []
    bounds = [] # for instrument gap

    # Load channels
    for p in [ri_p, yj_p, h_p]:
        with fits.open(p) as hdul:
            flux = hdul[t].data
            if lin_scale: flux *= 1e19
            wav = hdul[9].data
            combined_lambda.append(wav)
            combined_flux.append(flux)
            bounds.append((wav.min(), wav.max()))

    # Flatten and Sort
    final_lambda = np.concatenate(combined_lambda) * u.AA
    final_flux = np.concatenate(combined_flux) * u.Unit('erg cm-2 s-1 AA-1')
    sort_idx = np.argsort(final_lambda)
    
    # Create Spectrum1D in rest-frame
    # lambda_rest = lambda_obs / (1+z)
    spec_rest = Spectrum(
        spectral_axis=(final_lambda[sort_idx] / (1 + redshift)),
        flux=final_flux[sort_idx]
    )

    # REBIN to common grid
    rebinned_spec = resampler(spec_rest, common_grid)
    flux_vals = rebinned_spec.flux.value # Convert to raw numpy array

    # Create a mask for intrument gap: Start with all False (gap)
    valid_mask = np.zeros(len(common_grid), dtype=bool)
    for obs_min, obs_max in bounds:
        # De-redshift the boundaries of each arm
        rest_min = obs_min / (1 + redshift)
        rest_max = obs_max / (1 + redshift)
        
        # Mark pixels in the common_grid that fall within a real detector arm
        valid_mask |= (common_grid.value >= rest_min) & (common_grid.value <= rest_max)

    # Set everything outside the arms (the gaps) to 0.0
    flux_vals[~valid_mask] = 0.0
    # ---------------------------------

    # 4. INDIVIDUAL NORMALIZATION
    # We use the median of the non-zero pixels
    nonzero_pixels = flux_vals[valid_mask]
    norm_factor = 1.0
    if len(nonzero_pixels) > 0:
        norm_factor = np.median(nonzero_pixels)
        # Avoid division by zero if median is 0
        if norm_factor != 0:
            flux_vals = flux_vals / norm_factor


    # Update the plot to show the enforced gaps
    if base_name == valid_file_triplets[0][0].split('/')[-1].replace('_RI.fits', ''):
        plt.figure(figsize=(12, 6))
        plt.step(spec_rest.spectral_axis, spec_rest.flux, 
                where='mid', color='gray', alpha=0.3, label='Input (Rest)')
        plt.step(common_grid, flux_vals, 
                where='mid', color='crimson', lw=1.5, label='Rebinned & Masked')
        
        # Highlight where the gaps were enforced
        plt.fill_between(common_grid.value, 0, np.nanmax(flux_vals), 
                         where=~valid_mask, color='black', alpha=0.1, label='Enforced Gap')
        
        plt.title(f"Verification: {base_name} (z={redshift})")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{base_name}_check.png"))
        plt.show()
        plt.close()

    # --- SAVE TO FITS ---
    col1 = fits.Column(name='lambda', format='D', array=common_grid.value)
    col2 = fits.Column(name='flux', format='D', array=flux_vals) # Use the masked flux
    
    hdr = fits.Header()
    hdr['REDSHIFT'] = redshift
    hdr['ORIGINAL'] = base_name
    hdr['NORMFAC'] = norm_factor # Important to keep for science!
    
    hdu = fits.BinTableHDU.from_columns([col1, col2], header=hdr)
    
    out_name = f"{base_name}_{noise_type}_rebinned.fits"
    if lin_scale: out_name = out_name.replace('_rebinned', '_scaled_rebinned')
    
    hdu.writeto(os.path.join(output_dir, out_name), overwrite=True)


