from __future__ import annotations

import numpy as np
import pytest

from admet_uq.curves import EvalSink
from admet_uq.featurize import MORGAN_DIM
from admet_uq.heads import get_head
from admet_uq.heads import mlp as mlp_head
from admet_uq.storage import EvalRows, PredictionWriter, read_predictions

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
    """The MC-dropout arm."""
    return get_head("mlp_mcd")


def make_case(rng, task: str, n_train: int = 24, n_eval: int = 10):
    """A training block and the rows a fit is evaluated on."""
    n_features = MORGAN_DIM + 4
    x_train = rng.random((n_train, n_features), dtype=np.float32)
    if task == BINARY_TASK:
        y_train = (rng.random(n_train) > 0.5).astype(np.float32)
        y_eval = (rng.random(n_eval) > 0.5).astype(np.float64)
    else:
        y_train = rng.normal(0.0, 1.0, n_train).astype(np.float32)
        y_eval = rng.lognormal(0.0, 1.0, n_eval)
    rows = EvalRows(
        "inner",
        0,
        "val",
        np.arange(n_eval),
        np.array([f"c{i}" for i in range(n_eval)]),
        y_eval,
    )
    return x_train, y_train, rng.random((n_eval, n_features), dtype=np.float32), rows


def test_it_names_its_arm_and_its_scale(head):
    """The method key and the outputs are what the driver and the tables read."""
    assert (head.HEAD, head.UQ) == ("mlp", "mcd")
    assert head.OUTPUTS == ("mean", "std")
    assert head.SCALE_OUTPUTS == ("std",)
    assert head.POINT_OUTPUT == "mean"


def test_it_names_both_selection_rules(head, rng):
    """An arm carrying uncertainty is selected under the leaderboard metric and the likelihood."""
    _, _, _, rows = make_case(rng, BINARY_TASK)
    objectives = head.objectives(BINARY_TASK, rows)
    assert list(objectives) == ["leaderboard", "nll"]
    assert objectives["leaderboard"].direction == "maximize"
    assert objectives["nll"].direction == "minimize"


@pytest.mark.parametrize("task", [BINARY_TASK, LOG_TASK])
def test_it_records_a_mean_and_a_positive_spread(head, task, rng):
    """Every epoch carries both outputs, and the passes disagree."""
    x_train, y_train, x_eval, rows = make_case(rng, task)
    spy = SpySink(x_eval)

    head.fit(task, x_train, y_train, CONFIG, [spy], n_iterations=4)

    assert sorted(spy.recorded) == ["mean", "std"]
    assert spy.recorded["mean"].shape == (4, len(x_eval))
    assert spy.recorded["std"].shape == (4, len(x_eval))
    assert np.all(spy.recorded["std"] > 0)


def test_the_spread_stays_in_the_modelling_space(head, rng, tmp_path):
    """The mean is mapped to original units on the way to storage and the spread is not."""
    x_train, y_train, x_eval, rows = make_case(rng, LOG_TASK)
    path = tmp_path / "predictions.parquet"
    with PredictionWriter(path) as writer:
        sink = EvalSink(
            LOG_TASK, rows, "cfg", x_eval, writer=writer, scale_outputs=head.SCALE_OUTPUTS
        )
        head.fit(LOG_TASK, x_train, y_train, CONFIG, [sink], n_iterations=3)

    stored = read_predictions(path)
    means = np.stack(stored[stored["output"] == "mean"]["pred"].to_numpy())
    scales = np.stack(stored[stored["output"] == "std"]["pred"].to_numpy())
    # exp of a real number is positive, and a standard deviation is too, so the
    # test is that the spread is small where the exponentiated mean is not.
    assert np.all(means > 0)
    assert np.all(scales > 0)
    assert scales.max() < means.max()


def test_the_arm_trains_the_same_network_as_the_point_predictor(head, tmp_path, rng):
    """Drawing at evaluation moves no weight the deterministic arm would have reached.

    An arm takes its keys from the step after the epoch's last training step, so
    the training stream is the same one whether or not anything draws from it.
    Comparing the weights rather than the predictions is what makes this a
    statement about training.
    """
    deterministic = get_head("mlp")
    x_train, y_train, x_eval, rows = make_case(rng, BINARY_TASK)

    weights = []
    for name, module in (("point", deterministic), ("mcd", head)):
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

    assert sorted(weights[0]) == sorted(weights[1])
    for name in weights[0]:
        np.testing.assert_array_equal(np.asarray(weights[0][name]), np.asarray(weights[1][name]))


def test_the_mean_over_passes_tracks_the_point_prediction(head, rng):
    """Dropout is unbiased in expectation, so K passes centre near the deterministic pass."""
    deterministic = get_head("mlp")
    x_train, y_train, x_eval, rows = make_case(rng, LOG_TASK, n_train=64, n_eval=32)

    point, stochastic = SpySink(x_eval), SpySink(x_eval)
    deterministic.fit(LOG_TASK, x_train, y_train, CONFIG, [point], n_iterations=3, seed=1)
    head.fit(LOG_TASK, x_train, y_train, CONFIG, [stochastic], n_iterations=3, seed=1)

    spread = stochastic.recorded["std"][-1]
    gap = np.abs(stochastic.recorded["mean"][-1] - point.recorded["mean"][-1])
    # The K-pass mean is itself an average of K draws, so it sits well inside the
    # per-pass spread of the point prediction.
    assert np.median(gap) < np.median(spread)
