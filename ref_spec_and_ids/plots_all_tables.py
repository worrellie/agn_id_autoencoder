from astropy.io import fits

f = fits.open('ref_spec_and_ids/cosmos_bagpipes_202598_2h_z1.2546_H.fits')
f.info()
print()
for i, hdu in enumerate(f):
    print(f'=== HDU {i}: name={hdu.name!r}, type={type(hdu).__name__} ===')
    print(repr(hdu.header))
    print()

from astropy.io import fits
import numpy as np

f = fits.open('ref_spec_and_ids/cosmos_bagpipes_202598_2h_z1.2546_H.fits')
h = f[1].header
n, crval, cdelt, crpix = h['NAXIS1'], h['CRVAL1'], h['CDELT1'], h['CRPIX1']

# FITS WCS is 1-indexed: lambda(i) = CRVAL1 + (i+1 - CRPIX1)*CDELT1, i = 0..n-1
i = np.arange(n)
wave = crval + (i + 1 - crpix) * cdelt

print('n          ', n)
print('CRVAL1     ', crval)
print('CDELT1     ', cdelt)
print('CRPIX1     ', crpix)
print('header WMIN/WMAX', f[0].header['WMIN'], f[0].header['WMAX'])
print('reconstructed first/last:', wave[0], wave[-1])

# compare with the explicit 'Vacuum wavelengths' image in HDU 9
vac = f[9].data
print('HDU9 vacuum first/last  :', vac[0], vac[-1])
print('max abs diff WCS vs HDU9:', float(np.max(np.abs(wave - vac))))

# === HDU 1: name='', type=ImageHDU ===
# XTENSION= 'IMAGE   '           / Image extension                                
# BITPIX  =                  -64 / array data type                                
# NAXIS   =                    1 / number of array dimensions                     
# NAXIS1  =                 3871                                                  
# PCOUNT  =                    0 / number of parameters                           
# GCOUNT  =                    1 / number of groups                               
# NAME    = 'Flux-calib frame'                                                    
# CRVAL1  =              14520.0                                                  
# CDELT1  =   0.8988391376451075                                                  
# CRPIX1  =                  1.0                                                  
# TUNIT1  = 'AA      '                                                            
# R       =               6700.0                                                  
# SAMPLING=                  2.7                                                  
# TUNIT2  = 'Angstrom-1 cm-2 erg s-1'  