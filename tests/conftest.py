import sys
import pathlib

# Ensure the project root is on the path so `import funcs`, `import training` etc. work
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
import numpy as np
import torch

BATCH = 4
INPUT_SIZE = 16
LATENT_SIZE = 4
# Minimal 1-hidden-layer config matching StandardAutoencoder / VAEAutoencoder constructor
CONFIG = [{"in": 8, "out": 8}]


@pytest.fixture
def batch_tensors():
    """Clean batch: x_hat ≈ x, full mask (no gaps)."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, INPUT_SIZE)
    x_hat = x + 0.01 * torch.randn(BATCH, INPUT_SIZE)
    x_mask = torch.ones(BATCH, INPUT_SIZE)
    return x, x_hat, x_mask


@pytest.fixture
def masked_batch_tensors():
    """Batch with first quarter of pixels masked to zero."""
    torch.manual_seed(1)
    x = torch.randn(BATCH, INPUT_SIZE)
    x_hat = x + 0.05 * torch.randn(BATCH, INPUT_SIZE)
    x_mask = torch.ones(BATCH, INPUT_SIZE)
    x_mask[:, : INPUT_SIZE // 4] = 0.0
    return x, x_hat, x_mask


@pytest.fixture
def vae_extras():
    """mu=0, logvar=0 → posterior == prior → KL == 0."""
    mu = torch.zeros(BATCH, LATENT_SIZE)
    logvar = torch.zeros(BATCH, LATENT_SIZE)
    return mu, logvar


@pytest.fixture
def norm_stats():
    """Simple normalization stats: mean=0.5, std=2.0, broadcastable with (BATCH, INPUT_SIZE)."""
    mean = torch.full((INPUT_SIZE,), 0.5)
    std = torch.full((INPUT_SIZE,), 2.0)
    return mean, std


@pytest.fixture
def sae_model():
    """Minimal StandardAutoencoder on CPU."""
    from autoencoder import StandardAutoencoder

    return StandardAutoencoder(
        CONFIG,
        input_size=INPUT_SIZE,
        latent_size=LATENT_SIZE,
        flux_type="normalized_flux_cont",
        normalize=False,
        activation="ReLU",
    )


@pytest.fixture
def vae_model():
    """Minimal VAEAutoencoder on CPU."""
    from autoencoder import VAEAutoencoder

    return VAEAutoencoder(
        CONFIG,
        input_size=INPUT_SIZE,
        latent_size=LATENT_SIZE,
        flux_type="normalized_flux_cont",
        normalize=False,
        activation="ReLU",
    )
