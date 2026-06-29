#!/bin/bash
#SBATCH --job-name=proc_spectra
#SBATCH --output=/home/vboyanov/ml_out/proc_%j.out
#SBATCH --error=/home/vboyanov/ml_out/proc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
# ---- Email Notifications ----
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# ---- environment setup ----
# Adjust this path to your actual working directory
RUNPATH=/home/vboyanov/ml/
cd $RUNPATH

# Activate your virtual environment
source $RUNPATH/bt_env/bin/activate

# Slurm automatically sets $SLURM_CPUS_PER_TASK based on the directive above,
# but we explicitly export it here just to be absolutely certain your python script sees it.
export SLURM_CPUS_PER_TASK=16

# Navigate to the folder containing your script, the "spectra/" dir, and "ref_spec_and_ids/"
# (Update this to wherever your script actually lives)
cd $RUNPATH/autoencoder/

# Run the python script
python process_spectra.py