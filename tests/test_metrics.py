from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from admet_uq.metrics import accuracy_metrics, aleatoric_floor, spearman

RNG = np.random.default_rng(0)


def _replicated(n_pairs: int, noise: float, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``n_pairs`` compounds measured twice, plus as many measured once."""
    means = RNG.normal(offset, 5.0, n_pairs)
    cid = np.repeat([f"pair{i}" for i in range(n_pairs)], 2)
    y = np.repeat(means, 2) + RNG.normal(0.0, noise, 2 * n_pairs)
    singles = RNG.normal(0.0, 5.0, n_pairs)
    return (
        np.concatenate([cid, [f"one{i}" for i in range(n_pairs)]]),
        np.concatenate([y, singles]),
    )


def test_pooled_sd_matches_hand_computation():
    """Two compounds, one pair each: the pooled sd is the RMS of the half-gaps."""
    cid = np.array(["a", "a", "b", "b", "c"])
    y = np.array([1.0, 3.0, 10.0, 10.0, 7.0])
    out = aleatoric_floor("ppbr_az", cid, y, min_dof=1)
    # deviations from the compound means are -1, +1, 0, 0 over dof = 2
    assert out["pooled_sd"] == pytest.approx(np.sqrt(2 / 2))
    assert out["dof"] == 2.0
    assert out["n_replicated"] == 2.0
    assert out["n_singleton"] == 1.0


def test_recovers_a_known_noise_scale():
    """Pooling over pairs estimates the scale the replicates were drawn with."""
    cid, y = _replicated(400, noise=2.0)
    out = aleatoric_floor("ppbr_az", cid, y)
    # each pair contributes one dof, so the relative standard error is 1/sqrt(2*400)
    assert out["pooled_sd"] == pytest.approx(2.0, rel=0.1)
    assert out["reportable"] == 1.0


def test_withheld_when_replicated_compounds_differ():
    """A replicated subset drawn from a shifted distribution fails the gate."""
    cid, y = _replicated(400, noise=2.0, offset=20.0)
    out = aleatoric_floor("ppbr_az", cid, y)
    assert out["representative_p"] < 0.05
    assert out["reportable"] == 0.0
    assert np.isfinite(out["pooled_sd"])


def test_withheld_when_too_few_replicates():
    """Enough dof, and the same gate passes; below ``min_dof`` it does not."""
    cid, y = _replicated(120, noise=2.0)
    assert aleatoric_floor("ppbr_az", cid, y, min_dof=100)["reportable"] == 1.0
    assert aleatoric_floor("ppbr_az", cid, y, min_dof=200)["reportable"] == 0.0


def test_no_replicates_reports_nothing():
    """A task measured once per compound yields no estimate and no error."""
    cid = np.array([f"c{i}" for i in range(50)])
    out = aleatoric_floor("ppbr_az", cid, RNG.normal(size=50))
    assert np.isnan(out["pooled_sd"])
    assert out["dof"] == 0.0
    assert out["reportable"] == 0.0


def test_binary_task_raises():
    """Replicate disagreement on a binary label is not a noise scale."""
    cid = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="binary"):
        aleatoric_floor("dili", cid, np.array([0.0, 1.0, 1.0, 1.0]))


def _one_value(value: float, n: int) -> np.ndarray:
    """``n`` copies of ``value``, a third of them one ulp above it.

    Per-compound averaging and a calibrator are the arithmetic paths that leave
    copies of a constant prediction differing in their last bits.
    """
    repeated = np.full(n, value)
    repeated[::3] = np.nextafter(value, np.inf)
    return repeated


def test_spearman_declines_one_repeated_value():
    """Ranks decided by last bits score nothing, where scipy returns a number for them."""
    y_true = RNG.normal(50.0, 10.0, 90)
    repeated = _one_value(91.6, 90)
    assert np.isfinite(spearmanr(y_true, repeated)[0])
    assert np.isnan(spearman(y_true, repeated))


def test_spearman_keeps_the_value_where_the_prediction_orders():
    """A prediction with a spread above the bound scores what scipy scores."""
    y_true = RNG.normal(50.0, 10.0, 90)
    predicted = y_true + RNG.normal(0.0, 5.0, 90)
    assert spearman(y_true, predicted) == spearmanr(y_true, predicted)[0]


def test_the_primary_metric_of_a_spearman_task_follows_it():
    """``primary`` is NaN wherever the Spearman it names is, and the error metrics stand."""
    compound_id = np.array([f"c{i:03d}" for i in range(60)])
    out = accuracy_metrics(
        "half_life_obach", RNG.lognormal(0.0, 1.0, 60), _one_value(3.4, 60), compound_id
    )
    assert np.isnan(out["spearman"])
    assert np.isnan(out["primary"])
    assert np.isfinite(out["mae"])
