import glob, os
import numpy as np
from astropy.io import fits

train_files = sorted(glob.glob("processed_spectra/*_rebinned.fits"))   # or your real train split

# replicate the pipeline's one-pass accumulation
n = 0; s = 0.0; s2 = 0.0
pooled = []                       # for the stable reference (fine at this data size)
for f in train_files:
    with fits.open(f) as hdul:
        raw = hdul[1].data["flux"].astype(np.float64)
    x = raw[raw != 0]
    n += x.size; s += x.sum(); s2 += np.sum(x**2)
    pooled.append(x)

onepass = np.sqrt(max(0, s2/n - (s/n)**2))
stable  = np.concatenate(pooled).std()     # all training pixels at once, stable
print(f"onepass = {onepass:.6e}")
print(f"stable  = {stable:.6e}")
print(f"rel diff = {abs(onepass - stable)/stable:.2e}")