import random
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits


# 1. Define files to ignore
exclude_files = {}

# 2. Filter FITS files using pathlib
current_dir = Path('/home/worrellie/Documents/phd/autoencoder/z09_and_z08')
# fits_files = [f for f in current_dir.glob("*.fits") if f.name not in exclude_files]
fits_files = ["mambo_13000110000176_z0.8005_1h_H.fits", "mambo_13000110000176_z0.8005_1h_RI.fits", "mambo_13000110000176_z0.8005_1h_YJ.fits"]

if not fits_files:
    print("No valid FITS files found.")
else:
    for f in fits_files:
        # random_file = random.choice(fits_files)
        print(f"Plotting: {f}")

        with fits.open(f'/home/worrellie/Documents/phd/autoencoder/z09_and_z08/{f}') as hdul:
            # Get title from Primary HDU (0)
            header_title = hdul[0].header.get('OBJECT', 'Spectrum Analysis')
            
            wavelengths = hdul[9].data
            
            # 3. Create a figure with 4 rows and 2 columns
            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), sharex=True)
            
            # Flatten the 2D array of axes into a 1D list for easy looping
            axes_flat = axes.flatten()

            titles = ["","HDU1: Mock spectrum with noise, in ergs/s/cm^2/AA","HDU2: template with sky, with no noise, in counts",
                    "HDU3: sky with no noise, in counts", "HDU4: input model template with no noise, in ergs/s/cm^2/AA",
                    "HDU5: sky flux with noise, in ph/s/cm^2/AA/arcsec^2", "HDU6: transmission curve, in ph/s/cm^2/AA/arcsec^2",
                    "HDU7: sky mask; values of 1 indicate masked pixels." , "HDU8: noise array, in ergs/s/cm^2/AA",
                    "HDU9: vacuum wavelength array, in AA"]

            # 4. Loop through HDUs 1 to 8
            for i in range(1, 9):
                ax = axes_flat[i-1] # Index 0 to 7
                spectrum_data = hdul[i].data
                if i == 1:
                    print(wavelengths, spectrum_data)
                    print(len(wavelengths))
                
                ax.plot(wavelengths, spectrum_data, color='midnightblue', lw=1)
                ax.set_title(f"{titles[i]}", fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Label only the bottom plots and left-most plots to save space
                if i > 6: ax.set_xlabel('Wavelength')
                if i % 2 != 0: ax.set_ylabel('Flux')

            # Add the Primary HDU data as a main super-title
            plt.suptitle(f"File: {f}", 
                        fontsize=16, fontweight='bold', y=0.95)

            # Adjust layout so titles don't overlap axes
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.show()