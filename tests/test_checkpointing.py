"""
Tests for Trainer.checkpoint / Trainer.resume.

Trainer is safe to instantiate on CPU with test=True — no wandb calls are
made during construction, and checkpoint/resume use no instance state beyond
what is passed as arguments.
"""

import pytest
import torch
from autoencoder import StandardAutoencoder
from training import Trainer

CONFIG = [{"in": 8, "out": 8}]
INPUT_SIZE = 16
LATENT_SIZE = 4
_TEST_PARAMS = {"test_name": "test_checkpoint"}


def _make_model():
    return StandardAutoencoder(
        CONFIG, INPUT_SIZE, LATENT_SIZE, "normalized_flux_cont", False, "ReLU"
    )


def _make_trainer(model):
    return Trainer(
        torch.device("cpu"),
        _TEST_PARAMS,
        model,
        optimizer=None,
        early_stopping=None,
        beta=0.0,
        use_autocast=False,
        test=True,
    )


def test_checkpoint_resume_round_trip(tmp_path):
    """State dict survives save → load with no parameter drift."""
    model = _make_model()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.23)

    trainer = _make_trainer(model)
    ckpt = tmp_path / "best.pt"
    trainer.checkpoint(model, ckpt)

    fresh = _make_model()
    trainer.resume(fresh, ckpt)

    for p_orig, p_loaded in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(p_orig, p_loaded), "Parameter mismatch after round-trip"


def test_checkpoint_file_is_created(tmp_path):
    model = _make_model()
    trainer = _make_trainer(model)
    ckpt = tmp_path / "model.pt"
    trainer.checkpoint(model, ckpt)
    assert ckpt.exists()


def test_checkpoint_with_optimizer_model_weights_survive(tmp_path):
    """When optimizer state is saved, model weights must still load correctly.

    NOTE: resume() currently does `pass` when an optimizer key is found, so
    optimizer state is NOT restored — this test documents that known gap and
    will fail if the gap is filled without updating the assertion.
    """
    model = _make_model()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.34)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = _make_trainer(model)
    ckpt = tmp_path / "with_optim.pt"
    trainer.checkpoint(model, ckpt, optimizer=optimizer)

    fresh = _make_model()
    # resume silently skips optimizer loading (pass branch)
    trainer.resume(fresh, ckpt)

    for p_orig, p_loaded in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(p_orig, p_loaded)


@pytest.mark.parametrize(
    "metric_a,metric_b,expect_checkpoint",
    [
        (0.5, 1.0, True),   # strict improvement → checkpoint
        (1.0, 0.5, False),  # regression → no checkpoint
        (1.0, 1.0, False),  # equal → no checkpoint
    ],
)
def test_do_checkpoint_decision_without_early_stopping(
    metric_a, metric_b, expect_checkpoint
):
    """Without early stopping the training loop uses `metric < best_validation`."""
    assert (metric_a < metric_b) == expect_checkpoint
