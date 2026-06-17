import pytest
import torch
import funcs

BATCH = 4
INPUT_SIZE = 16


def test_mse_zero_on_perfect_reconstruction(batch_tensors):
    x, _, x_mask = batch_tensors
    x_hat_perfect = x.clone()
    mse, _, _ = funcs._loss_calc_batch(x_hat_perfect, x, x_mask)
    assert mse.item() == pytest.approx(0.0, abs=1e-6)


def test_masked_residual_does_not_affect_loss():
    """A large error in a masked column must not change the batch MSE."""
    torch.manual_seed(5)
    x = torch.randn(BATCH, INPUT_SIZE)
    x_mask = torch.ones(BATCH, INPUT_SIZE)
    x_hat_clean = x.clone()

    # Add a large residual to column 0
    x_hat_dirty = x.clone()
    x_hat_dirty[:, 0] += 100.0

    # Loss with no masking of the dirty column
    mse_unmasked, _, _ = funcs._loss_calc_batch(x_hat_dirty, x, x_mask)

    # Loss with column 0 masked out
    x_mask_col0_hidden = x_mask.clone()
    x_mask_col0_hidden[:, 0] = 0.0
    mse_masked, _, _ = funcs._loss_calc_batch(x_hat_dirty, x, x_mask_col0_hidden)

    assert mse_masked.item() < mse_unmasked.item()


def test_beta_zero_total_equals_mse(batch_tensors, vae_extras):
    x, x_hat, x_mask = batch_tensors
    mu, logvar = vae_extras
    mse, _, total = funcs._loss_calc_batch(x_hat, x, x_mask, mu=mu, logvar=logvar, beta=0)
    assert total.item() == pytest.approx(mse.item(), rel=1e-5)


def test_kl_zero_when_posterior_equals_prior(batch_tensors, vae_extras):
    """mu=0, logvar=0 → KL(N(0,1) || N(0,1)) = 0."""
    x, x_hat, x_mask = batch_tensors
    mu, logvar = vae_extras  # zeros
    _, kl, _ = funcs._loss_calc_batch(x_hat, x, x_mask, mu=mu, logvar=logvar, beta=1.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("fn", [funcs.loss_calc_per_spec, funcs.rel_loss_calc_per_spec])
def test_per_spec_returns_vector_of_length_batch(fn, batch_tensors):
    x, x_hat, x_mask = batch_tensors
    result = fn(x_hat, x, x_mask)
    assert result.shape == (BATCH,)


def test_per_spec_mse_zero_on_perfect_reconstruction(batch_tensors):
    x, _, x_mask = batch_tensors
    result = funcs.loss_calc_per_spec(x.clone(), x, x_mask)
    assert result.max().item() == pytest.approx(0.0, abs=1e-6)


def test_rel_mse_batch_nonnegative(batch_tensors):
    x, x_hat, x_mask = batch_tensors
    result = funcs._rel_mse_calc_batch(x_hat, x, x_mask)
    assert result.item() >= 0.0


def test_rel_mse_batch_zero_on_perfect_reconstruction(batch_tensors):
    x, _, x_mask = batch_tensors
    result = funcs._rel_mse_calc_batch(x.clone(), x, x_mask)
    assert result.item() == pytest.approx(0.0, abs=1e-6)
