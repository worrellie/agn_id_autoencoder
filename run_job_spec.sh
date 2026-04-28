#!/bin/bash

#SBATCH --job-name=process_spec
#SBATCH --output=/home/vboyanov/ml_out/process_spec-%j.out
#SBATCH --error=/home/vboyanov/ml_out/process_spec-%j.err
#SBATCH --mail-user=valentinboyanov@tecnico.ulisboa.pt
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

RUNPATH=/home/vboyanov/ml
cd $RUNPATH

source $RUNPATH/bt_env/bin/activate

export SLURM_CPUS_PER_TASK=8

python -u process_galaxy_spectra.py