"""
Quick smoke test: train a tiny SAE on test_spec.h5 for a few epochs
with no wandb dependency, then check loss is finite and decreasing.
"""

import torch
import numpy as np
from torch import optim

import wandb
wandb.init(mode="disabled")   # suppress all wandb calls without a login

import funcs
import autoencoder as ae
import training
from datahandling import H5SpecDataset

# ── config ────────────────────────────────────────────────────────────────────
DATA        = "test_spec.h5"
FLUX_TYPE   = "normalized_flux_cont"
NORMALIZE   = False
CONFIG      = [{"in": 256, "out": 64}]
LATENT_SIZE = 16
EPOCHS      = 10
LR          = 1e-3
TEST_PARAMS = {"test_name": "smoke_test"}
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cpu")

train_ds = H5SpecDataset(DATA, split="train",      flux_type=FLUX_TYPE)
valid_ds = H5SpecDataset(DATA, split="validation", flux_type=FLUX_TYPE)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=1, shuffle=False)

INPUT_SIZE = train_ds.n_pixels
print(f"input size: {INPUT_SIZE}  |  train: {len(train_ds)}  |  valid: {len(valid_ds)}")

model = ae.StandardAutoencoder(
    CONFIG, INPUT_SIZE, LATENT_SIZE, FLUX_TYPE, NORMALIZE, activation="ReLU"
)

optimizer = optim.Adam(model.parameters(), lr=LR)

trainer = training.Trainer(
    device, TEST_PARAMS, model,
    optimizer=optimizer,
    early_stopping=None,
    beta=0.0,
    use_autocast=False,
    test=True,   # disables file I/O and wandb
)

model, best_model, losses = trainer.train_ae(
    EPOCHS, train_loader, valid_loader=valid_loader
)

# ── sanity checks ─────────────────────────────────────────────────────────────
train_losses = losses["train_total"]
valid_losses = losses["valid_total"]

print("\n── per-epoch losses ──────────────────────────────────────")
for i, (tr, va) in enumerate(zip(train_losses, valid_losses)):
    print(f"  epoch {i+1:2d}  train={tr:.6f}  valid={va:.6f}")

assert all(np.isfinite(train_losses)), "NaN/Inf in train losses"
assert all(np.isfinite(valid_losses)), "NaN/Inf in valid losses"
assert train_losses[-1] < train_losses[0], (
    f"training loss did not decrease: {train_losses[0]:.6f} → {train_losses[-1]:.6f}"
)
print(f"\ntraining loss decreased: {train_losses[0]:.6f} → {train_losses[-1]:.6f}  ✓")

# check predictions run without error
outputs = funcs.get_predictions(valid_loader, best_model, TEST_PARAMS, test=True)
assert len(outputs) == len(valid_ds)
loss_val = outputs[0]["loss_scaled"]
assert np.isfinite(loss_val), f"non-finite prediction loss: {loss_val}"
print(f"valid reconstruction loss (scaled): {loss_val:.6f}  ✓")

print("\nsmoke test passed.")
