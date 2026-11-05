from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from admet_uq.calibrate import (
    SEPARATED_FALLBACK,
    TEMPERATURE_BOUND_FALLBACK,
    TEMPERATURE_LOG_BOUND,
    calibrate_classification,
    fit_gaussian_scale,
    fit_scale_factor,
    fit_temperature,
    separates_at_half,
)


def _slice(rng: np.random.Generator, n: int, separation: float = 1.5):
    """A calibration slice with controllable class overlap."""
    y = (rng.random(n) < 0.4).astype(float)
    logit = rng.normal(separation * (2 * y - 1), 1.0)
    return 1.0 / (1.0 + np.exp(-logit)), y


def test_temperature_falls_back_on_a_separated_slice(rng):
    """Under separation the likelihood has no finite maximiser.

    Temperature scaling has no intercept, so its boundary is pinned at p = 0.5;
    with no wrongly sided point, more confidence always improves the likelihood.
    """
    prob = np.concatenate([np.linspace(0.55, 0.99, 30), np.linspace(0.01, 0.45, 30)])
    y = np.array([1.0] * 30 + [0.0] * 30)
    assert separates_at_half(prob, y)

    output = calibrate_classification("temperature", prob, y, prob)
    assert output.fallback == SEPARATED_FALLBACK
    assert output.scale == 1.0
    np.testing.assert_allclose(output.y_pred, prob), "a fallback must leave predictions untouched"


def test_separation_test_is_deliberately_non_strict():
    """A point at exactly p = 0.5 contributes a constant log 2 for every beta.

    It therefore does not arrest the runaway either (the quasi-complete
    separation case of Albert & Anderson 1984), so the test uses >= / <=.
    """
    assert separates_at_half(np.array([0.5, 0.5, 0.9, 0.1]), np.array([1.0, 0.0, 1.0, 0.0]))
    assert not separates_at_half(np.array([0.4, 0.6]), np.array([1.0, 0.0]))


def test_a_well_posed_temperature_fit_stays_inside_its_bound(rng):
    """A fit resting on the bound reports where the search stopped, not an estimate."""
    prob, y = _slice(rng, 200, separation=1.0)
    temperature = fit_temperature(prob, y)
    assert abs(np.log(temperature)) < TEMPERATURE_LOG_BOUND
    assert calibrate_classification("temperature", prob, y, prob).fallback == ""


def test_temperature_resting_on_its_bound_reports_a_fallback():
    """A slice whose base rate sits outside the probabilities drives T to the bound.

    Temperature divides the logit and cannot shift it, so the likelihood improves
    all the way to a flat prediction and has no interior optimum. Separation does
    not catch this one: every label is positive, so no point sits on the wrong
    side of 0.5 in the sense that test checks.
    """
    y = np.ones(40)
    prob = np.concatenate([np.full(36, 0.2), np.full(4, 0.7)])
    assert not separates_at_half(prob, y)

    output = calibrate_classification("temperature", prob, y, prob)
    assert output.fallback == TEMPERATURE_BOUND_FALLBACK
    assert np.log(output.scale) >= TEMPERATURE_LOG_BOUND - 1e-4
    # The bound is where every probability is within 1e-2 of a half, which is
    # what makes stopping there report the same numbers as going further.
    assert np.allclose(output.y_pred, 0.5, atol=1e-2)


def test_temperature_leaves_ranking_metrics_invariant(rng):
    """AUROC is invariant under a strictly monotone map, so a calibrator moving it is broken.

    Temperature scaling is a strictly monotone transformation.
    """
    prob, y = _slice(rng, 400, separation=1.2)
    output = calibrate_classification("temperature", prob, y, prob)
    assert roc_auc_score(y, output.y_pred) == pytest.approx(roc_auc_score(y, prob), abs=1e-9)


def test_none_calibrator_is_the_identity(rng):
    """The mandatory raw baseline must not touch the predictions."""
    prob, y = _slice(rng, 100)
    np.testing.assert_array_equal(calibrate_classification("none", prob, y, prob).y_pred, prob)


def test_unknown_calibrator_raises(rng):
    """An unrecognised calibrator name must fail loudly rather than fall through."""
    prob, y = _slice(rng, 50)
    with pytest.raises(ValueError, match="unknown classification calibrator"):
        calibrate_classification("beta", prob, y, prob)


def test_gaussian_scale_is_the_residual_rms(rng):
    """The MLE constant scale, the regression counterpart of temperature."""
    y = rng.normal(0.0, 3.0, size=500)
    prediction = np.zeros_like(y)
    assert fit_gaussian_scale(prediction, y) == pytest.approx(np.sqrt(np.mean(y**2)), rel=1e-12)


def test_the_scale_factor_recovers_a_known_inflation(rng):
    """A head whose spread is uniformly too sharp is corrected by one number."""
    n = 4000
    scale = rng.uniform(0.5, 2.0, n)
    calib_pred = rng.normal(0.0, 1.0, n)
    calib_y = calib_pred + rng.normal(0.0, 1.0, n) * scale * 3.0

    assert fit_scale_factor(calib_pred, calib_y, scale) == pytest.approx(3.0, rel=0.05)


def test_the_scale_factor_is_one_on_an_already_calibrated_scale(rng):
    """Nothing to correct leaves the head's own spread alone."""
    n = 4000
    scale = rng.uniform(0.5, 2.0, n)
    calib_pred = rng.normal(0.0, 1.0, n)
    calib_y = calib_pred + rng.normal(0.0, 1.0, n) * scale

    assert fit_scale_factor(calib_pred, calib_y, scale) == pytest.approx(1.0, rel=0.05)


def test_a_collapsed_scale_does_not_divide_by_zero(rng):
    """A row whose passes agreed exactly is floored rather than made infinite."""
    factor = fit_scale_factor(np.zeros(4), np.array([1.0, 1.0, 1.0, 1.0]), np.zeros(4))
    assert np.isfinite(factor)
    assert factor > 0
