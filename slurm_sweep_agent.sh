#!/bin/bash
#SBATCH --job-name=ae_sweep
#SBATCH --output=/home/vboyanov/ml_out/run_%A_%a.out
#SBATCH --error=/home/vboyanov/ml_out/run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ---- args ----
SWEEP_PATH="$1"
if [ -z "$SWEEP_PATH" ]; then
    echo "Error: no sweep path provided."
    echo "Usage: sbatch slurm_sweep_agent.sh <entity/project/sweep_id>"
    exit 1
fi

# ---- environment setup ----
RUNPATH=/home/vboyanov/ml/
cd $RUNPATH
source $RUNPATH/bt_env/bin/activate
export SLURM_CPUS_PER_TASK=8

cd $RUNPATH/autoencoder/

# ---- run ----
echo "Starting wandb agent for sweep: $SWEEP_PATH"
echo "SLURM job: $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"

wandb agent "$SWEEP_PATH" --count 1