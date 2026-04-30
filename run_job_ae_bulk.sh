#!/bin/bash

#SBATCH --job-name=run_sae
#SBATCH --output=/home/vboyanov/ml_out/run_ae-%A_%a.out
#SBATCH --error=/home/vboyanov/ml_out/run_ae-%A_%a.err
#SBATCH --mail-user=valentinboyanov@tecnico.ulisboa.pt
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2

RUNPATH=/home/vboyanov/ml/
cd $RUNPATH

source $RUNPATH/bt_env/bin/activate

export SLURM_CPUS_PER_TASK=8

cd $RUNPATH/autoencoder/

python -u run_ae.py --task_id $SLURM_ARRAY_TASK_ID