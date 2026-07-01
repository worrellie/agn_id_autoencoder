#!/bin/bash
#SBATCH --job-name=build_h5
#SBATCH --output=/home/vboyanov/ml_out/build_h5_%j.out
#SBATCH --error=/home/vboyanov/ml_out/build_h5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL

# ---- environment setup ----
RUNPATH=/home/vboyanov/ml/
cd $RUNPATH
source $RUNPATH/bt_env/bin/activate

cd $RUNPATH/autoencoder/

python save_h5.py