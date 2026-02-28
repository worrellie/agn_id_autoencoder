import os
import random
import matplotlib.pyplot as plt
from astropy.table import Table

def plot_random_fits_tables(folder_path, num_files=5):
    # 1. Filter for .fits files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    
    if not all_files:
        print("No .fits files found in the directory.")
        return

    num_files = min(len(all_files), num_files)
    selected_files = random.sample(all_files, num_files)
    
    # 2. Setup Plotting
    fig, axes = plt.subplots(num_files, 1, figsize=(8, 4 * num_files), sharex=False)
    if num_files == 1: axes = [axes]

    for i, filename in enumerate(selected_files):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 3. Read the FITS table
            # Astropy automatically finds the first extension containing a table
            t = Table.read(file_path)
            
            # Get column names (assuming there are at least 2)
            cols = t.colnames
            x_col, y_col = cols[0], cols[1]
            
            # 4. Plotting
            axes[i].plot(t[x_col], t[y_col], linestyle='-', alpha=0.7)
            axes[i].set_title(f"File: {filename}")
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_col)
            axes[i].grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            axes[i].set_title(f"Error: {filename}")

    plt.tight_layout()
    plt.show()

# Usage:
plot_random_fits_tables('merged_z09_v3-001')
plot_random_fits_tables('merged_z09_v3-001_noiseless')