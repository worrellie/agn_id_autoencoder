import sys
import pathlib

# Ensure the project root is on the path so `import funcs`, `import training` etc. work
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
import h5py

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

@pytest.fixture(scope="session")
def tiny_h5(tmp_path_factory):
    """Minimal file with the same schema as save_h5.py: f4, NaN gaps, train-split attrs."""
    p = tmp_path_factory.mktemp("data") / "tiny.h5"
    rng = np.random.default_rng(0)
    n, n_pix = 32, 64

    with h5py.File(p, "w") as hf:
        hf.attrs["wavelengths"] = np.linspace(4000, 9000, n_pix).astype("f4")

        for split in ("train", "validation", "test"):
            g = hf.create_group(split)
            flux = rng.normal(0.0, 1.0, size=(n, n_pix)).astype("f4")
            # two inter-band gaps, shifted per-spectrum (mimics de-redshifting)
            for i in range(n):
                s = 10 + (i % 5)
                flux[i, s:s+4]       = np.nan
                flux[i, s+30:s+34]   = np.nan
            g.create_dataset("log_scale_flux", data=flux, dtype="f4")
            g.create_dataset("raw_flux",       data=flux, dtype="f4")
            g.create_dataset("redshift", data=rng.uniform(0.5, 2.0, n).astype("f4"))
            g.create_dataset("SNR",      data=rng.uniform(3, 30, n).astype("f4"))

        finite = hf["train"]["log_scale_flux"][:]
        finite = finite[~np.isnan(finite)]
        hf.attrs["norm_mean_log"] = float(finite.mean())
        hf.attrs["norm_std_log"]  = float(finite.std())
        hf.attrs["raw_mean"]      = float(finite.mean())
        hf.attrs["raw_std"]       = float(finite.std())

    return str(p)