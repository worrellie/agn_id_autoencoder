import json
import numpy as np
from astropy.table import Table
from astropy.io import fits

# --- CONFIG: adjust to your data ---
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
OUT           = "ref_spec_and_ids/common_region.json"
# -----------------------------------

# 1. observed span of the full (stitched) spectrum, from ONE triplet
band_bounds = {}
for band, path in REF_TRIPLET.items():
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
z = np.asarray(Table.read(REF_CATALOGUE)[Z_COL], dtype=float)
z = z[np.isfinite(z) & (z >= Z_LO) & (z <= Z_HI)]
print(f"{z.size} galaxies in {Z_LO} <= z <= {Z_HI}")

# 3. deredshift each edge into the z=0.9 frame and intersect
#    lambda_target = lambda_obs * (1 + Z_TARGET) / (1 + z)
factor      = (1.0 + Z_TARGET) / (1.0 + z)   # one per galaxy
blue_edges  = obs_min * factor
red_edges   = obs_max * factor

print(blue_edges)
print(red_edges)

common_blue = float(blue_edges.max())        # set by the LOWEST z
common_red  = float(red_edges.min())         # set by the HIGHEST z
assert common_blue < common_red, "no overlap — check inputs"
print(f"common region (z={Z_TARGET} frame): {common_blue} - {common_red}")

# 4. write it out for the rest of the pipeline
region = {
    "z_target": Z_TARGET,
    "z_range": [Z_LO, Z_HI],
    "observed_min": obs_min, "observed_max": obs_max,
    "common_min": common_blue, "common_max": common_red,
    "n_galaxies_considered": int(z.size),
    "set_by_z_min": float(z.min()), "set_by_z_max": float(z.max()),
}

with open(OUT, "w") as f:
    json.dump(region, f, indent=2)
print(f"wrote {OUT}")