from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from specutils import Spectrum
from astropy import units as u
from astropy.visualization import quantity_support
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
import re
import os

def get_spec_obj(spec_path):

    try:
        with fits.open(spec_path) as hdul:
            flux = hdul[1].data * u.Unit('erg cm-2 s-1 AA-1')
            l = hdul[9].data * u.AA # Vacuum wavelengths

        spec = Spectrum(spectral_axis = l, flux = flux)
        print(spec)

        # fig, ax = plt.subplots()
        # ax.step(spec.spectral_axis, spec.flux)
        # plt.title('original: ' + spec_path)f = h.data

        return spec

    except Exception as e:
        print(f"Cant open {spec_path} ({e})")

def new_spec_grid(spec, new_grid=np.arange(6471,9342, 0.5)* u.AA, ):

    res = FluxConservingResampler(extrapolation_treatment='zero_fill')

    new_spec = res(spec, new_grid)
    print(new_spec)

    # fig_new, ax_new = plt.subplots()
    # ax_new.step(new_spec.spectral_axis, new_spec.flux)
    # plt.title('new')

    return new_spec

def save_spec(grid, spec, save_path):

    # data = np.array([grid, spec])
    data = [grid, spec]

    t = Table(data, names=('l', 'flux'))

    t.write(save_path, format = 'fits', overwrite = True)


def process_files(dir,save_dir):

    for f in os.listdir(dir):
        if "_RI.fits" in f:

            spec_path = os.path.join(dir, f)

            save_name = f.replace(".fits","_test.fits")
            save_path = os.path.join(save_dir, save_name)

            spec = get_spec_obj(spec_path)
            new_spec = new_spec_grid(spec)
            
            save_spec(new_spec.spectral_axis, new_spec.flux, save_path,)

def read_spec(spec_path):

    try:
        with fits.open(spec_path) as hdul:
            data = hdul[1].data
            l= data['l']
            flux = data['flux']

        return l, flux

    except Exception as e:
        print(f"Cant open {spec_path} ({e})")




# main #

test_dir = "/home/worrellie/Documents/phd/autoencoder/test"
test_save_dir = "/home/worrellie/Documents/phd/autoencoder/new_specs"

dir = "/home/worrellie/Documents/phd/autoencoder/Datasets/z09_v3-001"
save_dir = "/home/worrellie/Documents/phd/autoencoder/processed_z09"

process_files(dir, save_dir)

exit()

# og_path = "/home/worrellie/Documents/phd/autoencoder/test_test/mambo_13000111000147_z0.8009_1h_RI.fits"
# test_path = "/home/worrellie/Documents/phd/autoencoder/new_specs/mambo_13000111000147_z0.8009_1h_RI_test.fits"

# with fits.open(test_path) as hdul:

#     t_data= hdul[1].data

#     l_data = t_data['l']
#     flux_data = t_data['flux']
#     # print(l_data)
#     # print(flux_data)
    
    # feature_matrix = np.stack([l_data, flux_data], axis=1)

with fits.open(og_path) as hdul:

            flux = hdul[2].data
            l = hdul[9].data

fig = plt.subplots()
# plt.step(l_data, flux_data)
plt.step(l, flux)
plt.title('what the fuck')
plt.show()


