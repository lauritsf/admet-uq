from __future__ import annotations

import numpy as np
import pytest

from admet_uq.calibrate import fit_noise_floor, fit_scale_factor

RNG = np.random.default_rng(20260805)


def _slice(scale: np.ndarray, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Predictions at zero and labels drawn at ``sqrt(scale^2 + noise^2)``."""
    total = np.sqrt(scale**2 + noise**2)
    return np.zeros_like(scale), RNG.normal(0.0, total)


def test_recovers_a_known_noise_scale():
    """A spread that is pure epistemic plus a constant noise recovers the constant."""
    epistemic = RNG.uniform(0.05, 0.4, 200_000)
    prediction, label = _slice(epistemic, noise=1.5)
    assert fit_noise_floor(prediction, label, epistemic) == pytest.approx(1.5, rel=0.02)


def test_reports_no_floor_when_the_scale_is_already_right():
    """With nothing left to explain, the fitted floor is small against the scale."""
    epistemic = RNG.uniform(0.5, 2.0, 200_000)
    prediction, label = _slice(epistemic)
    assert fit_noise_floor(prediction, label, epistemic) < 0.1 * epistemic.mean()


def test_the_multiplier_cannot_do_what_the_floor_does():
    """A tiny spread plus real noise: the floor fits it, the multiplier distorts it.

    The multiplier is forced to inflate every row by the same large factor, so a
    row the model was confident about stays proportionally confident. The floor
    puts the same absolute noise on every row, which is what generated them.
    """
    epistemic = np.concatenate([np.full(100_000, 0.001), np.full(100_000, 0.05)])
    prediction, label = _slice(epistemic, noise=1.0)

    floor = fit_noise_floor(prediction, label, epistemic)
    factor = fit_scale_factor(prediction, label, epistemic)

    assert floor == pytest.approx(1.0, rel=0.02)
    floored = np.sqrt(epistemic**2 + floor**2)
    scaled = factor * epistemic
    # The floor is near-constant across the two groups, as the truth is; the
    # multiplier preserves their 50x ratio.
    assert floored.max() / floored.min() == pytest.approx(1.0, abs=0.01)
    assert scaled.max() / scaled.min() == pytest.approx(50.0, rel=1e-6)


def test_heterogeneous_scale_does_not_truncate_at_residual_rms():
    """The optimum may exceed the residual RMS for heterogeneous scales."""
    epistemic = np.array([0.001, 100.0])
    prediction = np.zeros(2)
    label = np.array([10.0, 0.0])
    residual_rms = float(np.sqrt(np.mean((label - prediction) ** 2)))

    floor = fit_noise_floor(prediction, label, epistemic)

    assert floor > residual_rms
    assert floor == pytest.approx(9.9513290248, rel=1e-6)
    assert floor <= np.max(np.abs(label - prediction))


def test_overdispersed_scale_takes_the_zero_boundary():
    """A supplied scale wider than every residual needs no added variance."""
    epistemic = np.full(4, 10.0)
    prediction = np.zeros(4)
    label = np.array([-1.0, -0.5, 0.5, 1.0])
    assert fit_noise_floor(prediction, label, epistemic) == 0.0


def test_global_fit_does_not_stop_at_the_wrong_local_minimum():
    """Heterogeneous scales can give the variance likelihood multiple minima."""
    scale_squared = np.array(
        [
            1.89040457e8,
            3.81886230e-6,
            5.55106738e-9,
            1.87181938,
            8.06013177e5,
            1.11632590e-2,
            7.15726376e6,
            8.45384704e-3,
        ]
    )
    residual_squared = np.array(
        [
            2.00378245e6,
            1.79921412e-8,
            4.65008097e-5,
            2.81612710e-8,
            6.19686908e4,
            2.46721892e-1,
            1.29469721e3,
            5.85620682e-6,
        ]
    )
    floor = fit_noise_floor(
        np.zeros_like(scale_squared),
        np.sqrt(residual_squared),
        np.sqrt(scale_squared),
    )
    assert floor == pytest.approx(np.sqrt(0.0423), rel=0.02)


def test_a_perfect_fit_asks_for_no_floor():
    """Zero residual leaves nothing to add."""
    epistemic = RNG.uniform(0.1, 0.5, 100)
    assert fit_noise_floor(np.zeros(100), np.zeros(100), epistemic) == 0.0
