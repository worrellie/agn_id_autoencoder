#!/bin/bash
#SBATCH --job-name=ae_timing
#SBATCH --output=/home/vboyanov/ml_out/timing_%j.out
#SBATCH --error=/home/vboyanov/ml_out/timing_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL

# ---- environment ----
RUNPATH=/home/vboyanov/ml/
cd $RUNPATH
source $RUNPATH/bt_env/bin/activate
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd $RUNPATH/autoencoder/

echo "=== job $SLURM_JOB_ID  started $(date) ==="
echo "cpus=$SLURM_CPUS_PER_TASK  mem=$SLURM_MEM_PER_NODE MB  node=$(hostname)"

# ---- one fixed training scenario ----
# small-ish config on purpose: enough to measure per-epoch cost without a huge wall.
# scale EPOCHS down to ~3-5 for a pure timing probe, then extrapolate.
/usr/bin/time -v python autoencoder_main.py \
    --filename all_spectra.h5 \
    --project_name timing_probe \
    --flux_type log_scale_flux \
    --epochs 5 \
    --latent 64 \
    --learn_rate 1e-5 \
    --weight_decay 1e-8 \
    --leaky \
    --layers-2 \
    --sae

echo "=== finished $(date) ==="
