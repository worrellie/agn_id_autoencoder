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

np.set_printoptions(threshold=np.inf)

#####################
input_dir = '/home/worrellie/Documents/phd/autoencoder/test_data_for_processing'
t = 1  # 1: noisy, 4: template
norm = False
lin_scale = False
exps = [1, 2, 4, 8]

# Define output directory
suffix = "linearly_scaled_" if lin_scale else ""
noise_type = "noisy" if t == 1 else "noiseless"
output_dir = f'test_processed_z09_z08_normalized_{suffix}{noise_type}'
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

##

resampler = FluxConservingResampler()

i = 0
for ri_p, yj_p, h_p, redshift in valid_file_triplets:
    base_name = os.path.basename(ri_p).replace('_RI.fits', '')
    
    combined_lambda = []
    combined_flux = []
    bounds = []

    for p in [ri_p, yj_p, h_p]:
        with fits.open(p) as hdul:
            flux = hdul[t].data
            if lin_scale: flux *= 1e19
            wav = hdul[9].data
            combined_lambda.append(wav)
            combined_flux.append(flux)
            bounds.append((wav.min(), wav.max()))

    # get mask from first spec only (so it is uniform)
    # (it should be th same everywhere anyway but por si las moscase)
    if i == 0:
        mask_min = bounds[1][1] + 1
        mask_max = bounds[2][0] - 1
        instrument_gap = np.arange(start = mask_min, stop= mask_max, step = 1)
        instrument_flux = [np.nan] * len(instrument_gap)
    # include gap 'data'
    combined_lambda.append(instrument_gap)
    combined_flux.append(instrument_flux)

    # sort
    final_lambda = np.concatenate(combined_lambda) 
    final_flux = np.concatenate(combined_flux)
    sort_idx = np.argsort(final_lambda)
    final_lambda = final_lambda[sort_idx]
    final_flux = final_flux[sort_idx]

    # get mask where instrument gap is (true where masking)
    instrument_mask = (final_flux == np.nan)

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

    if norm:
        # mean = valid_pixels.mean()
        # std = valid_pixels.std()
        # flux_vals_norm = (flux_vals - mean)/ std
        flux_val = flux_vals_norm
    

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
        # plt.savefig(os.path.join(output_dir, f"{base_name}_check.png"))
        # plt.show()
        # plt.close()



    # --- SAVE TO FITS ---
    # make masked pixels 0 instead 0
    flux_vals = np.nan_to_num(flux_vals, nan=0)
    print(len(common_grid.value))
    print(len(flux_vals))


    col1 = fits.Column(name='lambda', format='D', array=common_grid.value)
    col2 = fits.Column(name='flux', format='D', array=flux_vals) # Use the masked flux

    hdr = fits.Header()
    hdr['REDSHIFT'] = redshift
    hdr['ORIGINAL'] = base_name
    # hdr['NORMFAC'] = norm_factor # Important to keep for science!
    
    hdu = fits.BinTableHDU.from_columns([col1, col2], header=hdr)
    
    out_name = f"{base_name}_{noise_type}_rebinned.fits"
    if lin_scale: out_name = out_name.replace('_rebinned', '_scaled_rebinned')
    
    hdu.writeto(os.path.join(output_dir, out_name), overwrite=True)

    i += 1



