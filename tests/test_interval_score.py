from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from admet_uq.metrics import interval_score, interval_score_per_observation

RNG = np.random.default_rng(20260805)


def test_a_covered_observation_scores_the_width():
    """Inside the interval there is no penalty, so the score is the width."""
    y = np.array([0.0, 1.0, 2.0])
    lower, upper = np.array([-1.0, 0.5, 1.0]), np.array([1.0, 2.5, 3.0])
    assert interval_score_per_observation(y, lower, upper, 0.90) == pytest.approx(upper - lower)


def test_an_excluded_observation_pays_two_over_alpha():
    """The penalty is 2 / alpha times the distance to the nearer bound."""
    score = interval_score_per_observation(
        np.array([3.0, -2.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.90
    )
    # width 1, then 20 x 2 above the upper bound and 20 x 2 below the lower one.
    assert score == pytest.approx([1.0 + 40.0, 1.0 + 40.0])


def test_the_level_sets_the_penalty():
    """A tighter nominal level punishes the same miss more."""
    args = (np.array([2.0]), np.array([0.0]), np.array([1.0]))
    assert interval_score(*args, 0.50) == pytest.approx(1.0 + 4.0 * 1.0)
    assert interval_score(*args, 0.90) == pytest.approx(1.0 + 20.0 * 1.0)
    assert interval_score(*args, 0.99) == pytest.approx(1.0 + 200.0 * 1.0)


@pytest.mark.parametrize("level", [0.80, 0.90, 0.95])
def test_minimised_at_the_true_quantiles(level):
    """Propriety: the true central interval beats any rescaling of it."""
    y = RNG.normal(0.0, 1.0, 400_000)
    z = norm.ppf(0.5 + level / 2)
    truth = interval_score(y, np.full_like(y, -z), np.full_like(y, z), level)
    for factor in (0.5, 0.8, 0.95, 1.05, 1.25, 2.0):
        rescaled = interval_score(
            y, np.full_like(y, -z * factor), np.full_like(y, z * factor), level
        )
        assert rescaled > truth


def test_an_offset_interval_scores_worse_than_a_centred_one():
    """Shifting a correct-width interval off the mean costs, at equal width."""
    y = RNG.normal(0.0, 1.0, 200_000)
    z = norm.ppf(0.95)
    centred = interval_score(y, np.full_like(y, -z), np.full_like(y, z), 0.90)
    shifted = interval_score(y, np.full_like(y, -z + 0.5), np.full_like(y, z + 0.5), 0.90)
    assert shifted > centred


def test_the_mean_is_the_mean_of_the_per_observation_scores():
    """The two entry points agree, which is what lets the tables use either."""
    y = RNG.normal(0.0, 2.0, 500)
    lower, upper = y - RNG.uniform(0.1, 3.0, 500), y + RNG.uniform(0.1, 3.0, 500)
    lower[:50] += 5.0  # push some observations outside, so the penalty is exercised
    assert interval_score(y, lower, upper, 0.90) == pytest.approx(
        float(np.mean(interval_score_per_observation(y, lower, upper, 0.90)))
    )
