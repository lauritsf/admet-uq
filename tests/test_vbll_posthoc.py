from __future__ import annotations

import numpy as np
import pytest

from admet_uq.predictive import vbll_posthoc

RNG = np.random.default_rng(20260811)
WIDTH = 16
ROWS = 96


def _features() -> np.ndarray:
    """Frozen activations of a last hidden block, non-negative as a ReLU leaves them."""
    return np.maximum(RNG.normal(size=(ROWS, WIDTH)), 0.0).astype(np.float32)


def _regression_case() -> tuple[np.ndarray, np.ndarray]:
    features = _features()
    y = (features @ RNG.normal(size=WIDTH) + RNG.normal(scale=0.1, size=ROWS)).astype(np.float32)
    return features, y


def test_the_selected_pair_comes_from_the_declared_grids():
    """``head_fit`` records the grids, so a selection outside them would misreport."""
    features, y = _regression_case()
    best = vbll_posthoc.fit_layer(features, y, "regression")

    assert best["vbll_prior_scale"] in vbll_posthoc.PRIOR_GRID
    assert best["vbll_wishart_scale"] in vbll_posthoc.WISHART_GRID
    assert np.isfinite(best["negative_elbo"])


def test_a_fit_is_reproducible_from_its_seed():
    """The cell records one seed, so the layer it names has to be recoverable."""
    features, y = _regression_case()
    first = vbll_posthoc.fit_layer(features, y, "regression", seed=3)
    second = vbll_posthoc.fit_layer(features, y, "regression", seed=3)

    assert first["vbll_prior_scale"] == second["vbll_prior_scale"]
    assert first["vbll_wishart_scale"] == second["vbll_wishart_scale"]
    assert first["negative_elbo"] == pytest.approx(second["negative_elbo"], rel=1e-12)


def test_the_target_scale_shifts_the_location_and_only_scales_the_spread():
    """A spread carries no location, which is what ``scale_outputs`` records."""
    features, y = _regression_case()
    best = vbll_posthoc.fit_layer(features, y, "regression")

    plain = vbll_posthoc.predict(best["params"], features, "regression")
    scaled = vbll_posthoc.predict(
        best["params"], features, "regression", target_mean=5.0, target_std=3.0
    )

    np.testing.assert_allclose(scaled["mean"], plain["mean"] * 3.0 + 5.0, rtol=1e-5)
    np.testing.assert_allclose(scaled["std"], plain["std"] * 3.0, rtol=1e-5)


def test_a_binary_predictive_is_a_probability_with_a_spread():
    """The softmax draws are averaged, so both outputs live on the unit interval."""
    features = _features()
    y = (RNG.uniform(size=ROWS) < 0.4).astype(np.float32)
    best = vbll_posthoc.fit_layer(features, y, "binary")

    out = vbll_posthoc.predict(best["params"], features, "binary")

    assert out["mean"].shape == (ROWS,)
    assert np.all((out["mean"] > 0.0) & (out["mean"] < 1.0))
    assert np.all(out["std"] >= 0.0)
