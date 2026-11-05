from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from admet_uq.featurize import MORGAN_DIM
from admet_uq.heads import get_head
from admet_uq.heads import mlp as mlp_head
from admet_uq.predictive import laplace
from admet_uq.storage import EvalRows

BINARY_TASK = "dili"
LOG_TASK = "half_life_obach"

CONFIG = {
    "config_id": "cfg",
    "n_layers": 1,
    "hidden_dim": 16,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 8,
}


class SpySink:
    """Keeps every output a fit records, in the head's own space."""

    def __init__(self, features: np.ndarray):
        """Target the rows a head will predict."""
        self.features = features
        self.recorded: dict[str, np.ndarray] = {}

    def record(self, predictions: np.ndarray, output: str = "mean") -> None:
        """Take the head's predictions without inverting or storing them."""
        self.recorded[output] = np.asarray(predictions)


@pytest.fixture
def head():
    """The Laplace arm."""
    return get_head("mlp_laplace")


def make_case(rng, task: str, n_train: int = 24, n_eval: int = 10):
    """A training block and the rows a fit is evaluated on."""
    n_features = MORGAN_DIM + 4
    x_train = rng.random((n_train, n_features), dtype=np.float32)
    if task == BINARY_TASK:
        y_train = (rng.random(n_train) > 0.5).astype(np.float32)
    else:
        y_train = rng.normal(0.0, 1.0, n_train).astype(np.float32)
    return x_train, y_train, rng.random((n_eval, n_features), dtype=np.float32)


def test_it_names_its_arm_and_its_list(head):
    """Nothing about the approximation is searched, so it shares the deterministic configs."""
    assert (head.HEAD, head.UQ) == ("mlp", "laplace")
    assert head.OUTPUTS == ("mean", "std")
    assert head.SCALE_OUTPUTS == ("std",)
    assert head.CONFIG_LIST == "mlp"


def test_it_names_both_selection_rules(head):
    """An arm carrying uncertainty is selected under the leaderboard metric and the likelihood."""
    rows = EvalRows("inner", 0, "val", np.arange(3), np.array(["a", "b", "c"]), np.zeros(3))
    assert list(head.objectives(BINARY_TASK, rows)) == ["leaderboard", "nll"]


def test_the_eigenbasis_gives_the_same_variance_as_a_solve(rng):
    """The route through the eigendecomposition is the quadratic form it stands for."""
    n_train, width, n_eval = 20, 6, 5
    curvature = np.asarray(rng.random(n_train) + 0.5)
    # The bias column is the constant one, in the design and in the rows alike.
    design = np.column_stack([rng.normal(size=(n_train, width)), np.ones(n_train)])
    features = np.asarray(rng.normal(size=(n_eval, width)))
    rows = np.column_stack([features, np.ones(n_eval)])
    prior = 0.37

    curved = design.T @ (design * curvature[:, None])
    eigenvalues, basis = np.linalg.eigh(curved)
    through_basis = np.asarray(
        laplace._variance(
            jnp.asarray(basis), jnp.asarray(eigenvalues + prior), jnp.asarray(features)
        )
    )

    direct = np.einsum(
        "ij,ji->i", rows, np.linalg.solve(curved + prior * np.eye(width + 1), rows.T)
    )
    np.testing.assert_allclose(through_basis, direct, rtol=1e-4)


def test_the_prior_precision_is_the_grids_argmax(rng):
    """The marginal likelihood is compared over the grid and its best point is returned."""
    eigenvalues = jnp.asarray(rng.random(8) * 10.0)
    chosen = float(laplace._prior_precision(eigenvalues, 4.0, 8))
    assert chosen in {float(value) for value in laplace.PRIOR_GRID}

    # A larger weight norm asks for a looser prior, since the prior's penalty on
    # the weights is what its precision multiplies.
    assert float(laplace._prior_precision(eigenvalues, 400.0, 8)) <= chosen


def test_the_arm_trains_the_same_network_as_the_point_predictor(head, tmp_path, rng):
    """The posterior is fitted around weights the deterministic arm would have reached."""
    deterministic = get_head("mlp")
    x_train, y_train, x_eval = make_case(rng, BINARY_TASK)

    weights = []
    for name, module in (("point", deterministic), ("laplace", head)):
        directory = tmp_path / name
        module.fit(
            BINARY_TASK,
            x_train,
            y_train,
            CONFIG,
            [SpySink(x_eval)],
            n_iterations=4,
            seed=7,
            artifacts_dir=directory,
        )
        weights.append(mlp_head.load_params(directory / mlp_head.LAST_CHECKPOINT))

    for name in weights[0]:
        np.testing.assert_array_equal(np.asarray(weights[0][name]), np.asarray(weights[1][name]))


def test_the_regression_point_prediction_is_the_deterministic_one(head, rng):
    """A Gaussian centred on the trained weights leaves the location where it was.

    The exact pin lives on the weights, in the test above. Here the two arms
    reach their predictions through separately compiled programs, and which
    kernels XLA autotunes into each depends on what else has been compiled in
    the process, so the last bit of a float32 forward pass is not stable across
    them. The tolerance is that rounding, not a margin on the claim.
    """
    deterministic = get_head("mlp")
    x_train, y_train, x_eval = make_case(rng, LOG_TASK)

    point, approximated = SpySink(x_eval), SpySink(x_eval)
    deterministic.fit(LOG_TASK, x_train, y_train, CONFIG, [point], n_iterations=3, seed=1)
    head.fit(LOG_TASK, x_train, y_train, CONFIG, [approximated], n_iterations=3, seed=1)

    np.testing.assert_allclose(
        approximated.recorded["mean"], point.recorded["mean"], rtol=1e-5, atol=0.0
    )
    assert np.all(approximated.recorded["std"] > 0)


def test_the_binary_probability_is_pulled_towards_a_half(head, rng):
    """The probit approximation divides the logit by the scale it is uncertain on."""
    deterministic = get_head("mlp")
    x_train, y_train, x_eval = make_case(rng, BINARY_TASK)

    point, approximated = SpySink(x_eval), SpySink(x_eval)
    deterministic.fit(BINARY_TASK, x_train, y_train, CONFIG, [point], n_iterations=3, seed=1)
    head.fit(BINARY_TASK, x_train, y_train, CONFIG, [approximated], n_iterations=3, seed=1)

    moved = np.abs(approximated.recorded["mean"] - 0.5)
    original = np.abs(point.recorded["mean"] - 0.5)
    assert np.all(moved <= original + 1e-6)
    assert np.any(moved < original - 1e-6)


def test_the_regression_spread_covers_the_noise(head, rng):
    """The predictive variance is the weight uncertainty on top of the residual scale."""
    x_train, y_train, x_eval = make_case(rng, LOG_TASK, n_train=40)
    spy = SpySink(x_eval)

    head.fit(LOG_TASK, x_train, y_train, CONFIG, [spy], n_iterations=4)

    assert np.all(np.isfinite(spy.recorded["std"]))
    assert np.all(spy.recorded["std"] > 0)
