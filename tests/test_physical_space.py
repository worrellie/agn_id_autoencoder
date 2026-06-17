"""
Round-trip tests for funcs._to_physical_space.

Each test applies the forward transform (the same pipeline used in save_h5 +
training.py) to produce x_model, then calls _to_physical_space to invert it,
and asserts we recover x_start within floating-point tolerance.

Forward pipeline order:
  1. log step : x_log = sign(x) * log1p(|x|)    [if flux_type == "log_scale_flux"]
  2. Z-score  : x_model = (x_log - mean) / std   [if normalize == True]

_to_physical_space inverts in reverse order (Z-score first, then log).
"""

import pytest
import torch
import funcs

BATCH = 4
INPUT_SIZE = 16

MEAN = torch.full((INPUT_SIZE,), 0.5)
STD = torch.full((INPUT_SIZE,), 2.0)


def _apply_log(x):
    return torch.sign(x) * torch.log1p(x.abs())


def _apply_zscore(x):
    return (x - MEAN) / STD


@pytest.mark.parametrize(
    "normalize,flux_type,forward_fn",
    [
        (
            False,
            "normalized_flux_cont",
            lambda x: x.clone(),
        ),
        (
            True,
            "normalized_flux_cont",
            _apply_zscore,
        ),
        (
            False,
            "log_scale_flux",
            _apply_log,
        ),
        (
            True,
            "log_scale_flux",
            lambda x: _apply_zscore(_apply_log(x)),
        ),
    ],
    ids=[
        "no_normalize_no_log",
        "normalize_no_log",
        "no_normalize_log",
        "normalize_log",
    ],
)
def test_physical_space_round_trip(normalize, flux_type, forward_fn):
    torch.manual_seed(42)
    x_start = torch.randn(BATCH, INPUT_SIZE)

    x_model = forward_fn(x_start)
    result = funcs._to_physical_space(x_model, MEAN, STD, normalize, flux_type)

    assert torch.allclose(result, x_start, atol=1e-5), (
        f"Round-trip failed for normalize={normalize}, flux_type={flux_type}. "
        f"Max abs diff: {(result - x_start).abs().max().item():.2e}"
    )
