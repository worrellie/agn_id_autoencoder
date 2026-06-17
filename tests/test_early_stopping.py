import pytest
from training import CustomEarlyStopping

_TEST_PARAMS = {"test_name": "test_early_stopping"}


def _make_es(patience=3, delta=0.0):
    return CustomEarlyStopping(_TEST_PARAMS, patience=patience, delta=delta, test=True)


def test_new_best_is_true_on_init():
    es = _make_es()
    assert es.new_best is True


def test_first_loss_always_counts_as_improvement():
    es = _make_es()
    es.check_early_stop(1.0, model=None, epoch=0)
    assert es.new_best is True
    assert es.stop_training is False
    assert es.no_improve_count == 0


def test_improving_loss_resets_counter():
    es = _make_es()
    es.check_early_stop(1.0, model=None, epoch=0)
    es.check_early_stop(0.5, model=None, epoch=1)
    assert es.new_best is True
    assert es.no_improve_count == 0
    assert es.stop_training is False


def test_non_improving_loss_increments_counter():
    es = _make_es()
    es.check_early_stop(1.0, model=None, epoch=0)
    es.check_early_stop(1.1, model=None, epoch=1)
    assert es.new_best is False
    assert es.no_improve_count == 1


def test_patience_triggers_early_stop():
    patience = 3
    es = _make_es(patience=patience)
    es.check_early_stop(1.0, model=None, epoch=0)
    for i in range(patience):
        es.check_early_stop(1.5, model=None, epoch=i + 1)
    assert es.stop_training is True


def test_patience_not_triggered_one_step_before():
    patience = 3
    es = _make_es(patience=patience)
    es.check_early_stop(1.0, model=None, epoch=0)
    for i in range(patience - 1):
        es.check_early_stop(1.5, model=None, epoch=i + 1)
    assert es.stop_training is False


def test_improvement_after_streak_resets_stop():
    es = _make_es(patience=5)
    es.check_early_stop(1.0, model=None, epoch=0)
    es.check_early_stop(1.1, model=None, epoch=1)
    es.check_early_stop(1.2, model=None, epoch=2)
    # Large improvement
    es.check_early_stop(0.3, model=None, epoch=3)
    assert es.new_best is True
    assert es.no_improve_count == 0
    assert es.stop_training is False


def test_delta_small_improvement_counts_as_no_improve():
    """Improvement < delta should NOT reset the counter."""
    es = _make_es(delta=0.1)
    es.check_early_stop(1.0, model=None, epoch=0)
    # 1.0 - 0.05 = 0.95; improvement is 0.05 < delta=0.1 → no improvement
    es.check_early_stop(0.95, model=None, epoch=1)
    assert es.new_best is False
    assert es.no_improve_count == 1


def test_delta_large_improvement_counts_as_improve():
    """Improvement > delta should reset the counter."""
    es = _make_es(delta=0.1)
    es.check_early_stop(1.0, model=None, epoch=0)
    # 1.0 - 0.5 = 0.5; improvement is 0.5 > delta=0.1 → improvement
    es.check_early_stop(0.5, model=None, epoch=1)
    assert es.new_best is True
    assert es.no_improve_count == 0
