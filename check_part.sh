#!/bin/bash
#SBATCH --job-name=check_partial
#SBATCH --output=/home/vboyanov/ml_out/check_%j.out
#SBATCH --error=/home/vboyanov/ml_out/check_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL

RUNPATH=/home/vboyanov/ml/
cd $RUNPATH
source $RUNPATH/bt_env/bin/activate

cd $RUNPATH/autoencoder/

python check_partial_writes.py