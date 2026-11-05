from __future__ import annotations

import json

import numpy as np
import pytest

from admet_uq.featurize import MORGAN_DIM
from admet_uq.heads import mlp

BINARY_TASK = "dili"
LOG_TASK = "half_life_obach"
MAE_TASK = "caco2_wang"

CONFIG = {
    "n_layers": 1,
    "hidden_dim": 16,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 8,
}


class SpySink:
    """Collects what a head records, in place of the driver's sink."""

    def __init__(self, features: np.ndarray):
        """Stand in for one evaluation set over ``features``."""
        self.features = features
        self.chunks: list[np.ndarray] = []

    def record(self, predictions: np.ndarray, output: str = "mean") -> None:
        """Keep one run of iterations as it arrives."""
        self.chunks.append(np.asarray(predictions))


def make_inputs(task: str, rng, n_train: int = 32, n_eval: int = 12):
    """Training rows and one evaluation sink over the real feature width."""
    x_train = rng.random((n_train, MORGAN_DIM + 8), dtype=np.float32) * 4
    x_eval = rng.random((n_eval, MORGAN_DIM + 8), dtype=np.float32) * 4
    if task == BINARY_TASK:
        y = (rng.random(n_train) > 0.5).astype(np.float64)
    else:
        y = rng.lognormal(0.0, 1.0, n_train)
    return x_train, y, SpySink(x_eval)


def test_the_morgan_block_alone_is_compressed(rng):
    """The descriptor block is already CDF-normalised, so only the counts are transformed."""
    x = rng.random((4, MORGAN_DIM + 8), dtype=np.float32) * 10
    transformed = mlp._transform_features(x)
    np.testing.assert_allclose(transformed[:, :MORGAN_DIM], np.log1p(x[:, :MORGAN_DIM]), rtol=1e-6)
    np.testing.assert_array_equal(transformed[:, MORGAN_DIM:], x[:, MORGAN_DIM:])


def test_the_caller_s_features_are_left_alone(rng):
    """The sink holds a view into the cached feature matrix, which the head must not write to."""
    x = rng.random((4, MORGAN_DIM + 8), dtype=np.float32) * 10
    original = x.copy()
    mlp._transform_features(x)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("task", [BINARY_TASK, LOG_TASK])
def test_the_whole_trajectory_is_recorded_in_one_call(task, rng):
    """The scan accumulates every epoch, so the sink is handed all of them at once.

    One call rather than one per epoch is what keeps ``mae`` and ``rmse`` summing
    in the same order they do for the boosted head.
    """
    x_train, y, sink = make_inputs(task, rng)
    result = mlp.fit(task, x_train, y, CONFIG, [sink], n_iterations=4)

    assert len(sink.chunks) == 1
    assert sink.chunks[0].shape == (4, len(sink.features))
    assert result.n_iterations == 4


def test_each_sink_gets_only_its_own_rows(rng):
    """Evaluation sets are predicted together and split back out by sink."""
    x_train, y, first = make_inputs(BINARY_TASK, rng, n_eval=12)
    second = SpySink(rng.random((7, MORGAN_DIM + 8), dtype=np.float32) * 4)
    mlp.fit(BINARY_TASK, x_train, y, CONFIG, [first, second], n_iterations=3)

    assert first.chunks[0].shape == (3, 12)
    assert second.chunks[0].shape == (3, 7)


def test_padding_does_not_reach_the_recorded_rows(rng):
    """Rows are padded to fixed shapes, and every padded row is sliced away again."""
    x_train, y, sink = make_inputs(BINARY_TASK, rng, n_train=30, n_eval=5)
    result = mlp.fit(BINARY_TASK, x_train, y, CONFIG, [sink], n_iterations=2)

    # 30 rows at batch 8 pads to 32, and 5 evaluation rows pad to EVAL_BLOCK.
    assert result.metadata["train_rows_padded_to"] == 32
    assert result.metadata["eval_rows_padded_to"] == mlp.EVAL_BLOCK
    assert sink.chunks[0].shape == (2, 5)
    assert np.isfinite(sink.chunks[0]).all()


def test_binary_records_probabilities(rng):
    """The head's binary output space is the same as the boosted head's."""
    x_train, y, sink = make_inputs(BINARY_TASK, rng)
    mlp.fit(BINARY_TASK, x_train, y, CONFIG, [sink], n_iterations=2)
    recorded = np.concatenate(sink.chunks)
    assert np.all((recorded >= 0.0) & (recorded <= 1.0))


def test_regression_records_the_training_space(rng):
    """Standardization is inverted inside the head, so the driver's transform still applies."""
    x_train = rng.random((32, MORGAN_DIM + 8), dtype=np.float32)
    y = rng.normal(50.0, 0.001, 32)  # A near-constant target the network can reach.
    sink = SpySink(rng.random((6, MORGAN_DIM + 8), dtype=np.float32))
    mlp.fit(MAE_TASK, x_train, y, CONFIG, [sink], n_iterations=30)

    # Predictions land in the target's own units, not in standardized units.
    assert np.abs(np.concatenate(sink.chunks)[-1].mean() - 50.0) < 1.0


def test_a_constant_target_does_not_divide_by_zero(rng):
    """A cluster-grouped inner fold on a small task can carry one value throughout."""
    x_train, _, sink = make_inputs(MAE_TASK, rng)
    result = mlp.fit(MAE_TASK, x_train, np.full(32, 3.0), CONFIG, [sink], n_iterations=2)

    assert result.metadata["target_std"] == 1.0
    assert np.isfinite(np.concatenate(sink.chunks)).all()


def test_sweep_fit_does_not_write_checkpoints(rng):
    """A sweep fit does not write checkpoint files."""
    x_train, y, sink = make_inputs(BINARY_TASK, rng)
    mlp.fit(BINARY_TASK, x_train, y, CONFIG, [sink], n_iterations=2)  # no artifacts_dir


def test_the_deployed_epoch_is_checkpointed(tmp_path, rng):
    """The refit knows its deployed epoch up front, so the weights at it are saved as they pass."""
    x_train, y, sink = make_inputs(BINARY_TASK, rng)
    mlp.fit(
        BINARY_TASK,
        x_train,
        y,
        CONFIG,
        [sink],
        n_iterations=4,
        artifacts_dir=tmp_path,
        checkpoint_at=2,
    )

    deployed = mlp.load_params(tmp_path / mlp.DEPLOYED_CHECKPOINT)
    last = mlp.load_params(tmp_path / mlp.LAST_CHECKPOINT)
    assert deployed.keys() == last.keys()
    # Two more epochs of training separate them.
    assert any(not np.array_equal(deployed[key], last[key]) for key in deployed)

    written = json.loads((tmp_path / mlp.HEAD_CONFIG_FILE).read_text())
    assert written["hidden_dim"] == CONFIG["hidden_dim"]
    assert written["n_iterations"] == 4
    assert written["morgan_transform"] == "log1p"


def test_a_checkpoint_reproduces_the_epoch_it_was_taken_at(tmp_path, rng):
    """The one end-to-end numerical check available for a head with nothing to diff against.

    Reloading the weights and predicting the same rows must give back the row of
    the recorded trajectory that the deployed epoch wrote.
    """
    x_train, y, sink = make_inputs(BINARY_TASK, rng)
    result = mlp.fit(
        BINARY_TASK,
        x_train,
        y,
        CONFIG,
        [sink],
        n_iterations=6,
        artifacts_dir=tmp_path,
        checkpoint_at=3,
    )
    trajectory = sink.chunks[0]

    for name, iteration in ((mlp.DEPLOYED_CHECKPOINT, 3), (mlp.LAST_CHECKPOINT, 6)):
        params = mlp.load_params(tmp_path / name)
        again = mlp.predict(params, sink.features, CONFIG, BINARY_TASK, result.metadata)
        np.testing.assert_allclose(again, trajectory[iteration - 1], atol=1e-5)


def test_the_objective_follows_the_leaderboard_metric():
    """The loss is locked per task, the same rule the boosted head uses."""
    assert mlp._objective(BINARY_TASK) == "logistic"
    assert mlp._objective(MAE_TASK) == "l1"
    assert mlp._objective(LOG_TASK) == "l2"
