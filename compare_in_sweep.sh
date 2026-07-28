#!/bin/bash
cd ~/Documents/agn_id_autoencoder/sae_sweep_1

BEST=RUN_StandardAutoencoder_nl1_ls256_e300_ReLU_B0e+00_lr3e-05_wd1.390913335133999e-06_esFalse_nTrue_z3ujyl6a
DEEP=RUN_StandardAutoencoder_nl2_ls256_e300_Tanh_B0e+00_lr3e-05_wd7.430598639323284e-08_esFalse_nTrue_y5h25jgb
TINY=RUN_StandardAutoencoder_nl3_ls2_e300_LeakyReLU_B0e+00_lr2e-06_wd3.075186252796454e-07_esFalse_nTrue_ikjnmkm8

uv run python ../analyse.py "$BEST" "$DEEP" "$TINY"           > analyse.log  2>&1
uv run python ../analyse.py "$BEST" "$DEEP" "$TINY" --compare > compare.log 2>&1