"""
Tensor-shape contracts for StandardAutoencoder and VAEAutoencoder.
All tests run on CPU with small synthetic inputs — no GPU or training needed.

Note: VAEAutoencoder.encode() is not currently implemented (the method is
commented out in autoencoder.py), so only StandardAutoencoder.encode() is
tested here.
"""

import pytest
import torch
from autoencoder import StandardAutoencoder, VAEAutoencoder

BATCH = 4
INPUT_SIZE = 16
LATENT_SIZE = 4
CONFIG = [{"in": 8, "out": 8}]


@pytest.mark.parametrize("activation", ["ReLU", "Tanh", "LeakyReLU"])
def test_sae_forward_output_shape(activation):
    model = StandardAutoencoder(
        CONFIG, INPUT_SIZE, LATENT_SIZE, "normalized_flux_cont", False, activation
    )
    x = torch.randn(BATCH, INPUT_SIZE)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == (BATCH, INPUT_SIZE)
    assert mu is None
    assert logvar is None


@pytest.mark.parametrize("activation", ["ReLU", "Tanh", "LeakyReLU"])
def test_vae_forward_output_shape(activation):
    model = VAEAutoencoder(
        CONFIG, INPUT_SIZE, LATENT_SIZE, "normalized_flux_cont", False, activation
    )
    x = torch.randn(BATCH, INPUT_SIZE)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == (BATCH, INPUT_SIZE)
    assert mu.shape == (BATCH, LATENT_SIZE)
    assert logvar.shape == (BATCH, LATENT_SIZE)


@pytest.mark.parametrize("activation", ["ReLU", "Tanh", "LeakyReLU"])
def test_sae_encode_shape(activation):
    model = StandardAutoencoder(
        CONFIG, INPUT_SIZE, LATENT_SIZE, "normalized_flux_cont", False, activation
    )
    x = torch.randn(BATCH, INPUT_SIZE)
    z = model.encode(x)
    assert z.shape == (BATCH, LATENT_SIZE)


def test_sae_forward_output_is_not_input():
    """Reconstruction should differ from input (non-trivial transform)."""
    model = StandardAutoencoder(
        CONFIG, INPUT_SIZE, LATENT_SIZE, "normalized_flux_cont", False, "ReLU"
    )
    x = torch.randn(BATCH, INPUT_SIZE)
    x_hat, _, _ = model(x)
    assert not torch.equal(x_hat, x)
