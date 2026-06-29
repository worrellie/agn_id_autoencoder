RND = 42

# from joblib import Parallel, delayed
from functools import partial
from functools import partial
import multiprocessing
import json
import glob
import concurrent
import numpy as np
from astropy.io import fits
# import re
# import joblib
# import random
from pathlib import Path
import sys

# from matplotlib import pyplot as plt

from specutils import Spectrum
import astropy.units as u
from specutils.manipulation import FluxConservingResampler

# from specutils import SpectralRegion

# from sklearn.model_selection import train_test_split

# import h5py
import os
# import random
from astropy.table import Table

# import warnings

def get_common_grid(ref_triplet, ref_catalogue, z_col = "TARGET_REDSHIFT", z_target = 0.9, z_range = [0.9, 1.7]):

    z_lo, z_hi = z_range[0], z_range[1]

    band_bounds = {}
    for band, path in ref_triplet.items():
        # wave = np.asarray(Table.read(path)[WAVE_COL], dtype=float)
        band_header = fits.getheader(path, ext=0)
        band_wmin = band_header["WMIN"]
        band_wmax = band_header["WMAX"]
        band_bounds[band] = (band_wmin, band_wmax)
        print(f"{band}: {band_wmin} - {band_wmax} - ({band_wmax - band_wmin} px)")

    obs_min = min(lo for lo, _ in band_bounds.values())
    obs_max = max(hi for _, hi in band_bounds.values())
    print(f"observed combined span: {obs_min} - {obs_max}")

    # 2. redshifts in range
    z = np.asarray(Table.read(ref_catalogue)[z_col], dtype=float)
    z = z[np.isfinite(z) & (z >= z_lo) & (z <= z_hi)]
    print(f"{z.size} galaxies in {z_lo} <= z <= {z_hi}")

    # 3. deredshift each edge into the z=0.9 frame and intersect
    #    lambda_target = lambda_obs * (1 + Z_TARGET) / (1 + z)
    factor      = (1.0 + z_target) / (1.0 + z)   # one per galaxy
    blue_edges  = obs_min * factor
    red_edges   = obs_max * factor

    common_blue = float(blue_edges.max())        # set by the LOWEST z
    common_red  = float(red_edges.min())         # set by the HIGHEST z
    assert common_blue < common_red, "no overlap — check inputs"
    print(f"common region (z={z_target} frame): {common_blue} - {common_red}")

    # 4. write it out for the rest of the pipeline
    region = {
        "z_target": z_target,
        "z_range": [z_lo, z_hi],
        "observed_min": obs_min, "observed_max": obs_max,
        "common_min": common_blue, "common_max": common_red,
        "n_galaxies_considered": int(z.size),
        "set_by_z_min": float(z.min()), "set_by_z_max": float(z.max()),
    }
    out = "./common_region.json"
    with open(out, "w") as f:
        json.dump(region, f, indent=2)
    print(f"wrote {out}")

    return region

def load_common_region(json_path):
    with open(json_path, "r") as f:
        region = json.load(f)
    return region

def check_common_region_exists(json_path):
    if not os.path.exists(json_path):
        # print(f"common region file {json_path} does not exist, creating it...")
        # find_common_grid(REF_TRIPLET, REF_CATALOGUE, z_col = Z_COL, z_target = Z_TARGET, z_range = [Z_LO, Z_HI])
        return False
    else:
        print(f"common region file {json_path} already exists, skipping creation.")
        return True
        
def get_valid_triplets(spec_dir):

    for z_dir in os.listdir(spec_dir):
        if not os.path.isdir(os.path.join(spec_dir, z_dir)):
            continue
        base_names = set()
        for s in os.listdir(os.path.join(spec_dir, z_dir)):
            # if not os.path.isdir(os.path.join(spec_dir, z_dir, d)):
            #     continue
            if not s.startswith("cosmos_bagpipes_"):
                continue
            base_name = s.split("_z")[0] # returns 'cosmos_bagpiped_id'
            z = s.split("_z")[1].split("_")[0] # returns 'redshift'
            z_float = float(z)
            if base_name in base_names:
                print(f"duplicate base name {base_name} in {z_dir})")
            else:
                base_names.add(base_name)
        for base_name in base_names:
            triplet = []
            for band in ["RI", "YJ", "H"]:
                expected_file = f"{base_name}_*_{band}.fits"
                matching = glob.glob(os.path.join(spec_dir, z_dir, expected_file))
                if not matching:
                    print(f"missing {band} for {base_name}, skipping")
                    break
                triplet.append(matching[0])
            else:
                yield triplet, z_float # makes a GENERATOR, gives back one triplet at 'pauses'


def get_channel_data(channel_path, t=1):

    # returns raw wavelength, flux and template of given channel
    # print(fits.getheader(channel_path))
    # exit()

    flux = fits.getdata(channel_path, ext=t)
    f_templ = fits.getdata(channel_path, ext=4)
    l = fits.getdata(channel_path, ext=9)

    return flux, l, f_templ

def deredshift_channel(flux, l, z, de_z=0.8):

    # returns de-redshifted wavelength and flux

    flux_z = flux * ((1 + z) / (1 + de_z))
    # flux_z = flux

    l_z = (1 + de_z) * (l / (1 + z))

    # gap = np.arange(, , , 0.5)
    # gap_de_z = (1 + de_z) * (l / (1 + z))

    # print(z)
    # print(flux_z, l_z)

    return flux_z, l_z

def rebin_channel(flux, l, resampler, grid_size=4.0):

    # resamples a channel to a common grid and creates a mask (unsure why i did this...)

    assert getattr(resampler, "extrapolation_treatment") == "truncate", (
        "Resampler must truncate values outside new grid"
    )

    master_grid = np.arange(0, 30000, grid_size)

    # shrink master grid to channel size for efficiency
    # *note* this also will crop up to 4AA, to make uniform
    mask = (master_grid >= np.min(l)) & (master_grid <= np.max(l))
    channel_grid = master_grid[mask]
    channel_grid = channel_grid * u.AA

    chan = Spectrum(flux * u.Unit("erg cm-2 s-1 AA-1"), l * u.AA)

    rebinned_chan = resampler(chan, channel_grid)

    # print(len(channel_grid))
    # print(len(l), len(rebinned_chan.spectral_axis.value))
    # print(len(flux), len(rebinned_chan.flux.value))
    # print(l, rebinned_chan.spectral_axis.value)
    # print(flux, rebinned_chan.flux.value)
    # print(min(rebinned_chan.spectral_axis), max(rebinned_chan.spectral_axis))
    # print(rebinned_chan.flux.value, rebinned_chan.spectral_axis.value)

    return rebinned_chan.flux.value, rebinned_chan.spectral_axis.value

def merge_channels(channel_pairs, grid_size):
    # combines all 3 channels and MASKS potentially questionable pixels with 0
    #

    full_l = np.arange(0, 30000, grid_size)

    full_flux = np.zeros_like(full_l)

    prev_end = None
    for f, l in channel_pairs:
        start_idx = int(np.round((l[0] - full_l[0]) / grid_size))
        end_idx = start_idx + len(f)

        # check overlap is not more than 1 or 2 pixels
        if prev_end is not None:
            overlap = start_idx - prev_end
            if 100 <= abs(overlap) <= 300:
                # print('instrument gap')
                pass
            elif -2 <= overlap <= 2:
                # print(f"expected gap")
                pass
            else:
                print(f"unexpected gap!! {overlap}")
        prev_end = end_idx

        # if masking edges with 0
        f_masked = f.copy()
        f_masked[0] = 0
        f_masked[-1] = 0

        full_flux[start_idx:end_idx] = f_masked

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

def calc_SNR(flux, l):

    flux = np.asarray(flux)
    l = np.asarray(l)

    target_mask = (l >= 5100) & (l <= 5800) & (flux != 0)

    target_flux = flux[target_mask]
    # print(len(target_flux))
    if target_flux.size < 2:
        print("could not get snr: continuum region too small")
        return 0.0, 0.0, 0.0

    noise = np.std(target_flux)
    mean_flux = np.mean(target_flux)
    # median_flux = np.median(target_flux)

    if noise == 0:
        print("could not get snr, zero noise")
        return mean_flux, 0.0, 0.0

    snr = mean_flux / noise

    return mean_flux, noise, snr

def save_spec( flux, l, original_z, snr, norm_factors, infile_base, outdir, noise_type="noisy"):

    col1 = fits.Column(name="lambda", format="D", array=l)
    col2 = fits.Column(name="flux", format="D", array=flux)

    hdr = fits.Header()
    hdr["OG_Z"] = str(original_z)
    hdr["SNR"] = str(snr)
    hdr["ORIGINAL"] = infile_base
    hdr["NORM_CON"] = str(norm_factors['continuum_mean'])
    hdr['NORM_MED'] = str(norm_factors['full_spec_median'])

    hdu = fits.BinTableHDU.from_columns([col1, col2], header=hdr)

    out_name = f"{infile_base}_{noise_type}_deZ_rebinned.fits"

    hdu.writeto(os.path.join(outdir, out_name), overwrite=True)

    return

def process_single_spec(triplet, common_vals,grid_size = 4.0, de_z = 0.9):

    resampler = FluxConservingResampler(extrapolation_treatment="truncate")

    t, redshift = triplet
    ri_p, yj_p, h_p = t[0], t[1], t[2]
    base_name = os.path.basename(ri_p).replace("_RI.fits", "")
    base_dir = Path(ri_p).parent.parent.parent  # parent of spectra/ dir (parent of parent of triplet)

    try:
        original_de_z_flux, original_de_z_l = [], []
        original_flux_rest, original_l_rest = [], []
        channel_pairs = []

        for channel in [ri_p, yj_p, h_p]:
            flux, l, _ = get_channel_data(
                channel
            )  # Assuming t=1 is hardcoded or parsed

            flux_de_z, l_de_z = deredshift_channel(flux, l, redshift, de_z=de_z)
            flux_rest, l_rest = deredshift_channel(flux, l, redshift, de_z=0.0)

            original_de_z_flux.extend(flux_de_z)
            original_de_z_l.extend(l_de_z)
            original_flux_rest.extend(flux_rest)
            original_l_rest.extend(l_rest)

            f_rebinned, l_rebinned = rebin_channel(
                flux_de_z, l_de_z, resampler, grid_size=grid_size
            )
            channel_pairs.append([f_rebinned, l_rebinned])

        # merge channels and get final spectrum
        spec_flux, spec_l = merge_channels(channel_pairs, grid_size=grid_size)
        final_spec_flux, final_spec_l = crop_spectrum(spec_flux, spec_l, common_vals)

        # check for fully masked spectra
        mask = final_spec_flux == 0
        if mask.all():
            print(f"fully masked spec: {base_name}")

        # get normalization factors
        cont_mean, noise, snr = calc_SNR( np.asarray(original_flux_rest), np.asarray(original_l_rest))
        if (noise == 0.0 and snr == 0.0) or (cont_mean == 0.0 or np.isnan(cont_mean) or cont_mean is None):
            print(f"invalid spec {base_name} with ({cont_mean}, {noise}, {snr})")


        full_spec_median = np.median(final_spec_flux[~mask])
        if  (full_spec_median == 0.0 or np.isnan(full_spec_median) or full_spec_median is None):
            print(f"invalid spec {base_name} with {full_spec_median}")
        
        norm_factors = {'continuum_mean' : cont_mean,
                        'full_spec_median' : full_spec_median}

        # save spectrum in fits
        output_dir = base_dir / "processed_spectra" # when var on left of / is pathlib.Path, / means to join paths
        output_dir.mkdir(parents=True, exist_ok=True) # create dir if it doesn't exist

        save_spec(final_spec_flux, final_spec_l, redshift, snr, norm_factors, base_name, output_dir,)

        return base_name  # Useful for tracking progress

    except Exception as e:
        print(
            f"\nERROR: Worker failed on file: {base_name}", file=sys.stderr, flush=True
        )
        print(f"Error details: {e}")
        # Re-raise the error if you want the whole job to stop,
        # or return None if you want the job to keep going for other files
        raise e


def main():

    REF_TRIPLET = {
    "RI": "ref_spec_and_ids/cosmos_bagpipes_202598_2h_z1.2546_RI.fits",
    "YJ": "ref_spec_and_ids/cosmos_bagpipes_202598_2h_z1.2546_YJ.fits",
    "H":  "ref_spec_and_ids/cosmos_bagpipes_202598_2h_z1.2546_H.fits",
    }
    # WAVE_COL      = "WAVE"        # wavelength column name in the spectra tables
    REF_CATALOGUE = "ref_spec_and_ids/bagpipes_cosmosID_xmatch.fits"
    Z_COL         = "TARGET_REDSHIFT"    # redshift column in the catalogue
    Z_TARGET      = 0.9           # frame to deredshift into
    Z_LO, Z_HI    = 0.9, 1.7      # only consider galaxies in this range
    # OUT           = "ref_spec_and_ids/common_region.json"

    if check_common_region_exists("common_region.json"):
        region = load_common_region("common_region.json")
    else:
        region = get_common_grid(REF_TRIPLET, REF_CATALOGUE, z_col = Z_COL, z_target = Z_TARGET, z_range = [Z_LO, Z_HI])

    common_vals = [region["common_min"], region["common_max"]]

    #####################################################################################################################
    
    if os.environ.get("SLURM_CPUS_PER_TASK") is not None:
        print("running on cluster")
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK"))
        print(f"Starting parallel processing on {cpus} cores...")
    else:
        print("running on non-cluster")
        cpus = multiprocessing.cpu_count() - 1  # Leave one core for the OS
        print(f"Starting parallel processing on {cpus} cores...")

    triplet_generator = get_valid_triplets("spectra")

    GRID_SIZE = 4.0  # Angstroms, for rebinning

    # notes for me: partial returns new function with some of the arguments 'frozen'/ already set for passing to executor
    # frozen args are the ones that every worker will use and will have the same.
    worker_function = partial(process_single_spec, common_vals = common_vals, grid_size = GRID_SIZE, de_z = Z_TARGET)

    # notes for me: execute extra processes. each extra process is a separate worker.
    # each separate worker is a separate python process, so they don't share memory.
    # executor manages the poool of workers- queues and hands out tasks.
    # worker chills until given task be executor then sends reuslts back and waits for next task
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        # executor.map applies the function to every item in the iterable (triplet_generator)
        # and returns an iterator of results
        # results come back in input order, not completion order. results are held back if the
        # first task takes longer than the second, for example.
        results = executor.map(worker_function, triplet_generator)

        for finished_base_name in results:
            if finished_base_name:
                print(f"Finished processing: {finished_base_name}")
                pass

    #####################################################################################################################

if __name__ == "__main__":

    main()