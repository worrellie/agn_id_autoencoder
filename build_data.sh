#!/bin/bash
#SBATCH --job-name=build_data
#SBATCH --output=/home/vboyanov/ml_out/build_data_%j.out
#SBATCH --error=/home/vboyanov/ml_out/build_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Stop at the first failing command, so the HDF5 step never runs on incomplete output.
set -euo pipefail

# ---- settings ----
RUNPATH=/home/vboyanov/ml
WORKDIR=$RUNPATH/autoencoder        # holds the scripts, spectra/, agn_spectra/, ref_spec_and_ids/

GALAXY_DIR=processed_spectra        # written by process_spectra.py (fixed in that script)
AGN_DIR=processed_agn_spectra
GALAXY_H5=all_spectra_float32_v4.h5
AGN_H5=all_agn_float32_v4.h5

# ---- environment ----
source $RUNPATH/bt_env/bin/activate
cd $WORKDIR
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, started $(date)"

# Existing processed spectra are skipped (logged as "exists") and not re-checked.
for d in $GALAXY_DIR $AGN_DIR; do
    if [ -d "$d" ] && [ -n "$(ls -A "$d")" ]; then
        echo "WARNING: $d already contains $(ls "$d" | wc -l) files; they will be kept, not reprocessed"
    fi
done

# ---- step 1: process spectra (galaxies, then AGN) ----
echo "===== step 1: process_spectra.py  $(date)"
python -u process_spectra.py galaxy agn

# ---- step 2: build HDF5 files ----
echo "===== step 2: save_h5.py  $(date)"
python -u save_h5.py --galaxy $GALAXY_DIR $GALAXY_H5 \
                     --agn    $AGN_DIR    $AGN_H5

echo "===== done  $(date)"
