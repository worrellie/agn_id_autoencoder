#!/bin/bash
# Submit one wandb sweep agent as a SLURM job.
#
# Usage:
#   1. Create a sweep and note the sweep ID:
#        wandb sweep sweep_config.yaml
#      This prints: "wandb: Created sweep with ID: <SWEEP_ID>"
#      Full sweep path is: worrellie-iastro/<project>/<SWEEP_ID>
#
#   2. Submit N parallel agents (each runs one trial):
#        sbatch --array=1-20%8 slurm_sweep_agent.sh worrellie-iastro/autoencoder_sweep/<SWEEP_ID>
#      %8 limits max concurrent jobs to 8 — adjust for your cluster.
#
#   Or submit a single job:
#        sbatch slurm_sweep_agent.sh worrellie-iastro/autoencoder_sweep/<SWEEP_ID>

#SBATCH --job-name=ae_sweep
#SBATCH --output=sweep_logs/run_%A_%a.out
#SBATCH --error=sweep_logs/run_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# Uncomment the line below if your cluster has GPUs:
##SBATCH --gres=gpu:1
# Set the partition for your cluster:
##SBATCH --partition=short

# ---- environment setup (adjust for your cluster) ----
# module load python/3.10
# source /path/to/venv/bin/activate
# conda activate your_env_name

# ---- run ----
SWEEP_PATH="$1"

if [ -z "$SWEEP_PATH" ]; then
    echo "Error: no sweep path provided."
    echo "Usage: sbatch slurm_sweep_agent.sh <entity/project/sweep_id>"
    exit 1
fi

mkdir -p sweep_logs

cd "$SLURM_SUBMIT_DIR"

echo "Starting wandb agent for sweep: $SWEEP_PATH"
echo "SLURM job: $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"

wandb agent "$SWEEP_PATH" --count 1
