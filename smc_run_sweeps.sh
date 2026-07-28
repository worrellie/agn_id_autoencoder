#!/bin/bash

# open venv:
source .venv/bin/activate


# make wandb config for standard AE and store the wandb command for the sweep:
# send output of run wandb sweep from where it usual goes (stream 2/ err) to stream 1/ out
# pipe | carries the stream 1 output to whatever (in this case grep)
# use $() to make variable
SAE_RUN_CMD=$(uv run wandb sweep sweep_config_sae.yaml 2>&1 | grep -oP 'wandb: Run sweep agent with: \s*\K.*')

# run the sae sweep:
# use {} to reference variable
uv run ${SAE_RUN_CMD}

VAE_RUN_CMD=$(uv run wandb sweep sweep_config_vae.yaml 2>&1 | grep -oP 'wandb: Run sweep agent with: \s*\K.*')
uv run $(VAE_RUN_CMD)

# remember to make it executable:
# chmod u+x smc_run_sweep.sh