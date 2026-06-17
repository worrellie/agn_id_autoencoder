# Galaxy Spectra Autoencoder

[![Tests](https://github.com/worrellie/autoencoder/actions/workflows/tests.yml/badge.svg)](https://github.com/worrellie/autoencoder/actions/workflows/tests.yml)

PyTorch autoencoder for reconstructing and encoding galaxy spectra. Supports standard and variational architectures, four flux normalisation strategies, Weights & Biases experiment tracking, and Bayesian hyperparameter sweeps via W&B.

---

## Overview

The pipeline has two stages:

1. **Pre-processing** (`process_galaxy_spectra.py`) — reads raw FITS spectra (three-channel: RI/YJ/H), de-redshifts, rebins to a common wavelength grid, merges channels, and writes an HDF5 file with train/validation/test splits and per-spectrum normalisation factors.

2. **Training** (`run_ae.py`) — loads the HDF5 file, trains a `StandardAutoencoder` or `VAEAutoencoder`, logs metrics and plots to W&B, and saves the best model.

---

## Project structure

```
autoencoder/
├── run_ae.py                  # training entry point
├── autoencoder.py             # StandardAutoencoder, VAEAutoencoder
├── training.py                # Trainer, CustomEarlyStopping
├── funcs.py                   # loss functions, physical-space inversion, statistics
├── datahandling.py            # H5SpecDataset (PyTorch Dataset)
├── plotting.py                # loss curves, distributions, example reconstructions
├── process_galaxy_spectra.py  # data pre-processing entry point
├── funcs_process_gals.py      # de-redshifting, rebinning, SNR, HDF5 writing
├── evaluate_ae.py             # evaluate a saved model on the test set
├── sweep_config.yaml          # W&B Bayesian sweep configuration
├── slurm_sweep_agent.sh       # submit sweep agents as SLURM jobs
├── pyproject.toml
└── tests/                     # pytest suite (54 tests, CPU-only, no real data)
```

---

## Installation

Requires Python ≥ 3.11. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone and set up the environment
git clone https://github.com/worrellie/autoencoder.git
cd autoencoder
uv sync
```

For development (adds pytest):

```bash
uv sync --group dev
```

---

## Data preparation

Raw FITS spectra must be organised as triplets: `<name>_<exptime>h_z<redshift>_RI.fits`, `_YJ.fits`, `_H.fits`.

```bash
uv run python process_galaxy_spectra.py \
    --input_dir /path/to/fits/ \
    --output_dir /path/to/processed/ \
    --h5_file all_spectra.h5
```

This produces an HDF5 file with `train`, `validation`, and `test` groups, each containing `log_scale_flux`, `normalized_flux_cont`, `normalized_flux_med`, and `raw_flux` datasets, plus global mean/std attributes computed from the training set only (unmasked pixels only, to avoid data leakage).

---

## Training

```bash
uv run python run_ae.py \
    -f all_spectra.h5 \
    -p my_project \
    --layers-1 \
    -l 32 \
    -e 100 \
    -ft log_scale_flux \
    -n \
    -s
```

Key arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `-f` | HDF5 data file | `all_spectra.h5` |
| `-p` | W&B project name | `unspecified_project` |
| `--layers-1` / `--layers-2` / `--layers-3` / `--layers-4` | Architecture depth | `--layers-1` |
| `-l` | Latent dimension | `32` |
| `-e` | Max epochs | `10` |
| `-ft` | Flux type: `log_scale_flux`, `normalized_flux_cont`, `normalized_flux_med`, `raw_flux` | `log_scale_flux` |
| `-n` | Apply runtime Z-score normalisation | off |
| `-s` | Enable early stopping | off |
| `-b` | KL weight β (VAE only) | `0.0` |
| `--vae` | Train a VAE instead of SAE | off |
| `-r` / `-t` / `--leaky` | Activation: ReLU / Tanh / LeakyReLU | ReLU |

The best model (by unscaled relative MSE on validation) is saved to `RUN_<name>/`.

---

## Hyperparameter sweeps (W&B)

```bash
# 1. create the sweep — note the printed sweep ID
wandb sweep sweep_config.yaml

# 2. run agents locally
wandb agent worrellie-iastro/autoencoder_sweep/<SWEEP_ID>
```

`sweep_config.yaml` runs Bayesian optimisation over latent size, learning rate, weight decay, number of layers, and activation function. The sweep metric is `final/best_valid_unscaled_rel_mse` (lower is better).

---

## Running on a SLURM cluster

```bash
# submit 20 parallel sweep agents (max 8 running at once)
sbatch --array=1-20%8 slurm_sweep_agent.sh worrellie-iastro/autoencoder_sweep/<SWEEP_ID>
```

Edit `slurm_sweep_agent.sh` to set your cluster's partition and activate the correct environment before submitting.

---

## Evaluation

```bash
uv run python evaluate_ae.py \
    -f all_spectra.h5 \
    -m RUN_<name>/<name>_best_model.pt \
    -p my_project
```

---

## Tests

The test suite covers pure-logic functions only — no GPU, no HDF5 files, no W&B. All tests use small synthetic numpy/torch arrays.

```bash
uv run pytest tests/ -v
```

54 tests across 7 modules:

| File | What is tested |
|------|---------------|
| `test_loss_funcs.py` | masked MSE, KL divergence, per-spectrum shape contracts |
| `test_physical_space.py` | encode→decode round-trip for all 4 normalisation×flux-type combinations |
| `test_early_stopping.py` | counter, patience, delta threshold state machine |
| `test_model_shapes.py` | SAE/VAE forward and encode output shapes, 3 activations |
| `test_normalisation.py` | zero-pixel preservation, NORMFAC computed from unmasked pixels only |
| `test_checkpointing.py` | checkpoint/resume file round-trip, do_checkpoint decision logic |
| `test_gal_processing.py` | de-redshift formula, SNR calculation, channel merging |

CI runs on every push via GitHub Actions.
