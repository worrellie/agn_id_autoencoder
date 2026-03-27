
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
import random
from astropy.table import Table

# np.set_printoptions(threshold=np.inf)


def get_common_grid(input_dir, exps = [1, 2, 4, 8], de_z = 0.8):

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
                    
                    # de-redshift to certain redshift (0.8 default)
                    all_rest_mins.append((1 + de_z) * obs_min / (1 + z))
                    all_rest_maxs.append((1 + de_z) * obs_max / (1 + z))
                    valid_file_triplets.append((ri_path, yj_path, h_path, z))
                except Exception as e:
                    print(f"Error reading {ri_path}: {e}")


    # common overlap when de-redshifted to 0.8
    common_min = np.max(all_rest_mins)
    common_max = np.min(all_rest_maxs)
    # common wavelength grid at 0.8
    # common_grid = np.arange(np.ceil(common_min), np.floor(common_max), 4.0) * u.AA

    print(f"common range: {common_min:.2f} to {common_max:.2f} Å")
    print(f"num specs to process: {len(valid_file_triplets)}")

    return [common_min, common_max], valid_file_triplets

def get_channel_data(channel_path, t = 1):

    flux = fits.getdata(channel_path, ext = t)
    f_templ = fits.getdata(channel_path, ext = 4)
    l = fits.getdata(channel_path, ext = 9)

    # with fits.open(channel_path) as hdul:
    #     flux = hdul[t].data
    #     f_templ = hdul[4].data
    #     l = hdul[9].data
    
    # print(flux, l)

    return flux, l, f_templ

def deredshift_channel(flux, l, z, de_z=0.8):

    flux_z = flux * ((1 + z)/(1 + de_z))

    l_z = (1 + de_z) * (l / (1 + z))

    gap = np.arange(, , , 0.5)
    gap_de_z = (1 + de_z) * (l / (1 + z))

    # print(z)
    # print(flux_z, l_z)

    return flux_z, l_z

def rebin_channel(flux, l, resampler,):

    assert getattr(resampler, 'extrapolation_treatment') == 'truncate', "Resampler must truncate values outside new grid"

    master_grid = np.arange(0, 30000, 4.0) 



    # min_l = min(l)
    # max_l = max(l)
    # # print(min_l, max_l)

    # channel_min = np.ceil(min_l)
    # channel_max = np.floor(max_l)

    mask = (master_grid >= np.min(l)) & (master_grid <= np.max(l))
    channel_grid = master_grid[mask] * u.AA

    chan = Spectrum(flux * u.Unit('erg cm-2 s-1 AA-1'), l * u.AA)

    rebinned_chan = resampler(chan, channel_grid)

    # print(len(channel_grid))
    # print(len(l), len(rebinned_chan.spectral_axis.value))
    # print(len(flux), len(rebinned_chan.flux.value))
    # print(l, rebinned_chan.spectral_axis.value)
    # print(flux, rebinned_chan.flux.value)
    # print(min(rebinned_chan.spectral_axis), max(rebinned_chan.spectral_axis))
    # print(rebinned_chan.flux.value, rebinned_chan.spectral_axis.value)

    return [rebinned_chan.flux.value, rebinned_chan.spectral_axis.value]


def merge_channels(channel_pairs):

    grid_step = 4.0

    full_l = np.arange(0, 30000, grid_step)

    full_flux =np.zeros_like(full_l)
    
    for f, l in channel_pairs:
        # idxs = np.where(np.isclose(full_l, l))[0]
        # print(idxs)
        # idxs = np.searchsorted(full_l, l)
        # full_flux[idxs] = f

        float_idxs = (l-full_l[0]) / grid_step
        round_idxs = np.round(float_idxs).astype(int)

        drift = np.abs(float_idxs - round_idxs)
        if np.any(drift > 0.01):
            print(f"Warning: Grid alignment drift detected! Max drift: {np.max(drift):.4f} bins")

        # 3. Assign only valid in-bounds indices
        valid = (round_idxs >= 0) & (round_idxs < len(full_l))
        full_flux[round_idxs[valid]] = f[valid]

    # print(len(full_l))
    # print(full_flux)
    # print(full_l)

    # print(type(full_flux))

    return full_flux, full_l

def crop_spectrum(flux, l, common_vals):
    
    common_min = np.ceil(common_vals[0])
    common_max = np.floor(common_vals[1])

    idxs = np.where((l >= common_min) & (l <= common_max))
    # print(len(idxs[0]))
    # print(idxs)
    cropped_flux = flux[idxs]
    cropped_l = l[idxs]

    # print(len(cropped_l ))
    # print(min(cropped_l))
    # print(max(cropped_l))

    return cropped_flux, cropped_l

def merge_orignal_de_z(flux, l):

    flux = np.asarray(flux)
    l = np.asarray(l)

    sort_idx = np.argsort(l)
    l = l[sort_idx]
    flux = flux[sort_idx]

    # spec = Spectrum(flux * u.Unit('erg cm-2 s-1 AA-1'), l * u.AA)

    return [flux, l]

def plot_specs(flux, l, original, templ = None):

    print()

    plt.figure(figsize = (12,6))
    plt.plot(original[1], original[0], color='black')
    plt.step(l, flux, color = 'red')
    if templ is not None:
        plt.plot(templ[1], templ[0], color='blue')
    plt.show()

def make_mask():



    return

################
##### main #####
################

# setup

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

# run code

common_vals, valid_triplets = get_common_grid(input_dir)

resampler = FluxConservingResampler(extrapolation_treatment = 'truncate')

for ri_p, yj_p, h_p, redshift in valid_triplets:

    base_name = os.path.basename(ri_p).replace('_RI.fits', '')

    original_de_z_flux = []
    original_de_z_f_templ = []
    original_de_z_l = []
    channel_pairs = []
    
    for channel in [ri_p, yj_p, h_p]:

        flux, l, f_templ = get_channel_data(channel)

        flux_de_z, l_de_z = deredshift_channel(flux, l, redshift)
        f_templ_de_z, _ = deredshift_channel(f_templ, l, redshift)

        original_de_z_flux.extend(flux_de_z)
        original_de_z_f_templ.extend(f_templ_de_z)
        original_de_z_l.extend(l_de_z)

        chan_pair = rebin_channel(flux_de_z, l_de_z, resampler)
        
        channel_pairs.append(chan_pair)

        print("---------------------")

    
    original = merge_orignal_de_z(original_de_z_flux, original_de_z_l)
    template = merge_orignal_de_z(original_de_z_f_templ, original_de_z_l)

    spec_flux, spec_l = merge_channels(channel_pairs)

    plot_specs(spec_flux, spec_l, original, templ = template)

    final_spec_flux, final_spec_l = crop_spectrum(spec_flux, spec_l, common_vals)



