from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from specutils import Spectrum
from astropy import units as u
from astropy.visualization import quantity_support
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
import re
import os

CHANNELS = ["RI", "YJ", "H"]

H_CHAN = "/home/worrellie/Documents/phd/autoencoder/Datasets/test/mambo_13000110000176_z0.8005_1h_H.fits"
RI_CHAN = "/home/worrellie/Documents/phd/autoencoder/Datasets/test/mambo_13000110000176_z0.8005_1h_RI.fits"
YJ_CHAN = "/home/worrellie/Documents/phd/autoencoder/Datasets/test/mambo_13000110000176_z0.8005_1h_YJ.fits"

MOCK_SPECTRUM_HDU = 1

def fits_info(path):

    with fits.open(path) as hdul:

        print(hdul.info())
        # print(hdul)
        print("+" * 50)

        # print(fits.PrimaryHDU().header)

        for h in hdul:
            head = h.header
            # print(head)
            print("+" * 50)
            if isinstance(h, fits.ImageHDU):
                f = h.data
                l = get_lambda(head)
                print(head)
                print(head["NAME"])
                # print(head["CRPIX1"])
                print(head["CRVAL1"])
                print(head["CDELT1"])
                fig, ax = plt.subplots()
                ax.step(l, f)
                plt.title(head['NAME'])
                plt.show()

def get_z(path):

    z_match = re.search(r'_z(.*?)_', path)

    try:
        z = float(z_match.group(1))
        return z
    except:
        print('Could not get redshift: ', path)
        return

def get_lambda(head):

        w = WCS(head)

        pixel_indices = np.arange(head["NAXIS1"])

        wavelengths = w.pixel_to_world(pixel_indices)

        return wavelengths

def get_spec(hdu, ):

    spec = hdu.data

    return spec

def plot_channel(l, spec):

    plt.plot(l, spec)

def get_channel(path, hdu_num = MOCK_SPECTRUM_HDU):

    with fits.open(path) as hdul:

        h = hdul[hdu_num]

        head = h.header

        print(head["CRVAL1"])
        print(head["CDELT1"])

        l = get_lambda(head)

        spec = get_spec(h)
    
    plt.plot(l,spec)
    plt.show()

    return l, spec


def get_spec_obj(path):

    with fits.open(path) as hdul:
        spec = hdul[MOCK_SPECTRUM_HDU]
        l = get_lambda(spec.header) * u.AA
        flux = spec.data * u.Unit('erg cm-2 s-1 AA-1')

    spec_obj = Spectrum(spectral_axis=l, flux=flux)

    # f, ax = plt.subplots()
    # ax.step(spec_obj.spectral_axis, spec_obj.flux)

    # plt.show()

    return l, spec_obj

def new_spec_grid(spec, new_grid=np.arange(0,15000, 0.5)* u.AA, ):

    res = FluxConservingResampler(extrapolation_treatment='zero_fill')

    new_spec = res(spec, new_grid)


    # f, ax = plt.subplots()
    # ax.step(new_spec.spectral_axis, new_spec.flux)

    # plt.show()
    return new_spec, new_grid

######### main 

path = "/home/worrellie/Documents/phd/autoencoder/test/mambo_13000110000176_z0.8005_1h_RI.fits"
fits_info(path)
exit()


with fits.open(path) as hdul:
    flux = hdul[1].data * u.Unit('erg cm-2 s-1 AA-1')
    l = hdul[9].data  * u.AA
    print(flux)
    print(l)
print(min(l), max(l))
print(min(flux),max(flux))
fig = plt.subplots()
# plt.step(l_data, flux_data)
plt.step(l, flux)
plt.title('what the fuck')
plt.show()




exit()
ls = []
spec_objs = []
new_spec_objs = []
for c in CHANNELS:
    if c == "RI":
    
        path = f"/home/worrellie/Documents/phd/autoencoder/test/mambo_13000111000147_z0.8009_1h_{c}.fits"
        l, spec_obj = get_spec_obj(path)
        # if c == "H":
        #     max_l = 
        # elif c == "RI":
        #     min_l = 
        ls.append(l)
        spec_objs.append(spec_obj)

        new_spec_obj, new_grid = new_spec_grid(spec_obj, )
        new_spec_objs.append(new_spec_obj)
        print(new_spec_obj)

final_spec = new_spec_objs[0]# + new_spec_objs[1] + new_spec_objs[2]
# print(final_spec)

f, ax = plt.subplots()
for s,c in zip(new_spec_objs,CHANNELS):
    ax.step(s.spectral_axis, s.flux, label =c)
# ax.step(final_spec.spectral_axis, final_spec.flux, where='mid', c='k', lw=1)
plt.legend

plt.show()

    
