from __future__ import annotations

import numpy as np
import pytest

from admet_uq import storage
from admet_uq.curves import leaderboard_objective
from admet_uq.heads import DEFAULT_SELECTION, DETERMINISTIC, FitResult, get_head
from admet_uq.nested_cv import run_cell
from admet_uq.splits import OuterFold, SplitAssignment

TASK = "dili"
N_ITERATIONS = 5
N_ROWS = 40


@pytest.fixture
def cell_inputs(rng):
    """A two-config, two-inner-fold cell over synthetic rows."""
    compound_id = np.array([f"C{i:03d}" for i in range(N_ROWS)])
    features = {
        "X": rng.random((N_ROWS, 6), dtype=np.float32),
        "y": (rng.random(N_ROWS) > 0.4).astype(np.float32),
        "compound_id": compound_id,
    }
    index = np.arange(N_ROWS)
    test_idx, calibration_idx, fit_idx = index[:10], index[10:18], index[18:]
    fold = OuterFold(
        train_idx=index[10:],
        test_idx=test_idx,
        calibration_idx=calibration_idx,
        fit_idx=fit_idx,
        inner=[
            (fit_idx[:11], fit_idx[11:]),
            (fit_idx[11:], fit_idx[:11]),
        ],
    )
    assignment = SplitAssignment(
        task=TASK,
        compound_ids=list(compound_id),
        cluster=np.zeros(N_ROWS, dtype=int),
        outer=np.zeros(N_ROWS, dtype=int),
        inner=np.zeros((N_ROWS, 1), dtype=int),
        n_outer=1,
        n_inner=2,
        seed=0,
        cutoff=0.5,
        calibration_fraction=0.2,
        rdkit_version="test",
        compound_digest="digest",
    )
    configs = [{"config_id": "stub_000"}, {"config_id": "stub_001"}]
    return features, assignment, [fold], configs


class StubHead:
    """A head that records a fixed trajectory and counts how it was called."""

    HEAD = "stub"
    UQ = DETERMINISTIC
    DEFAULT_ITERATIONS = N_ITERATIONS
    OUTPUTS = ("mean",)
    SCALE_OUTPUTS = ()
    POINT_OUTPUT = "mean"
    CONFIG_LIST = "stub"

    @staticmethod
    def objectives(task, rows):
        """Select on the task's leaderboard metric, as the real heads do."""
        return {DEFAULT_SELECTION: leaderboard_objective(task, rows)}

    def __init__(self, chunk: int = N_ITERATIONS):
        """Record in runs of ``chunk`` iterations."""
        self.chunk = chunk
        self.records: list[int] = []
        self.artifacts_dirs: list = []
        self.checkpoints: list = []

    def fit(
        self,
        task,
        x_train,
        y_train,
        config,
        eval_sinks,
        *,
        seed=0,
        n_jobs=0,
        n_iterations=DEFAULT_ITERATIONS,
        artifacts_dir=None,
        checkpoint_at=None,
    ):
        """Record a rising trajectory for each sink, in runs of ``chunk`` iterations."""
        self.artifacts_dirs.append(artifacts_dir)
        self.checkpoints.append(checkpoint_at)
        for sink in eval_sinks:
            calls = 0
            for start in range(0, n_iterations, self.chunk):
                size = min(self.chunk, n_iterations - start)
                step = np.arange(start, start + size, dtype=float)[:, None]
                sink.record(np.tile(sink.features[:, :1].T, (size, 1)) + step)
                calls += 1
            self.records.append(calls)
        return FitResult(n_iterations, {"stub": True})


def run(cell_inputs, tmp_path, head: StubHead, monkeypatch):
    """Run one cell against ``head`` and write its artifacts."""
    monkeypatch.setattr("admet_uq.nested_cv.get_head", lambda name: head)
    features, assignment, folds, configs = cell_inputs
    outputs = run_cell(
        TASK, "stub", 0, features, assignment, folds, configs, tmp_path, n_iterations=N_ITERATIONS
    )
    storage.write_curves(tmp_path / storage.CURVES_FILE, outputs.curve_records)
    storage.write_eval_rows(tmp_path / storage.EVAL_ROWS_FILE, outputs.eval_rows)
    storage.write_metadata(tmp_path / storage.METADATA_FILE, outputs.metadata)
    return outputs


def test_cell_writes_four_artifacts(cell_inputs, tmp_path, monkeypatch, rng):
    """A completed cell leaves predictions, curves, alignment rows and metadata."""
    outputs = run(cell_inputs, tmp_path, StubHead(), monkeypatch)

    for name in (
        storage.PREDICTIONS_FILE,
        storage.CURVES_FILE,
        storage.EVAL_ROWS_FILE,
        storage.METADATA_FILE,
    ):
        assert (tmp_path / name).is_file()

    # Two configs over two inner folds, plus the refit's calibration and test sets.
    predictions = storage.read_predictions(tmp_path / storage.PREDICTIONS_FILE)
    assert len(predictions) == (2 * 2 + 2) * N_ITERATIONS
    assert set(predictions["eval_set"]) == {"val", "calib", "test"}
    assert sorted(predictions["iteration"].unique()) == list(range(1, N_ITERATIONS + 1))

    curves = storage.read_curves(tmp_path / storage.CURVES_FILE)
    assert len(curves) == 2 * 2 * N_ITERATIONS
    assert outputs.metadata["n_iterations"] == N_ITERATIONS
    assert outputs.metadata["head_fit"] == {"stub": True}


def test_selected_iteration_is_within_the_budget(cell_inputs, tmp_path, monkeypatch, rng):
    """Selection reads the mean inner curve, so it can only land inside the budget."""
    outputs = run(cell_inputs, tmp_path, StubHead(), monkeypatch)
    assert 1 <= outputs.metadata["selected_iteration"] <= N_ITERATIONS
    assert outputs.metadata["deployed_iteration"] == outputs.metadata["selected_iteration"]
    assert outputs.metadata["selected_config_id"] in {"stub_000", "stub_001"}


def test_the_refit_alone_is_given_a_place_for_weights(cell_inputs, tmp_path, monkeypatch, rng):
    """Only the deployed fit is asked to checkpoint; the sweep's fits are not."""
    head = StubHead()
    outputs = run(cell_inputs, tmp_path, head, monkeypatch)

    sweep, refit = head.artifacts_dirs[:-1], head.artifacts_dirs[-1]
    assert all(directory is None for directory in sweep)
    assert refit == tmp_path
    assert head.checkpoints[:-1] == [None] * len(sweep)
    assert head.checkpoints[-1] == outputs.metadata["selected_iteration"]


def test_a_head_recording_per_iteration_gives_the_same_cell(
    cell_inputs, tmp_path, monkeypatch, rng
):
    """Chunking is the head's business; the stored rows and the selection do not move."""
    whole = run(cell_inputs, tmp_path / "whole", StubHead(chunk=N_ITERATIONS), monkeypatch)
    stepwise = run(cell_inputs, tmp_path / "stepwise", StubHead(chunk=1), monkeypatch)

    assert whole.metadata["selected_config_id"] == stepwise.metadata["selected_config_id"]
    assert whole.metadata["selected_iteration"] == stepwise.metadata["selected_iteration"]
    assert whole.metadata["test_metrics"] == stepwise.metadata["test_metrics"]

    left = storage.read_predictions(tmp_path / "whole" / storage.PREDICTIONS_FILE)
    right = storage.read_predictions(tmp_path / "stepwise" / storage.PREDICTIONS_FILE)
    keys = ["stage", "config_id", "inner_fold", "eval_set", "iteration"]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    for column in keys:
        np.testing.assert_array_equal(left[column], right[column])
    # The evaluation sets differ in length, so compare the vectors end to end.
    np.testing.assert_array_equal(
        np.concatenate(left["pred"].to_numpy()), np.concatenate(right["pred"].to_numpy())
    )


def test_unknown_head_is_rejected(cell_inputs, tmp_path):
    """A head name with no module behind it fails before any fitting."""
    features, assignment, folds, configs = cell_inputs
    with pytest.raises(ValueError, match="unknown head"):
        run_cell("dili", "nope", 0, features, assignment, folds, configs, tmp_path)


def test_lgbm_records_each_sink_once(cell_inputs, tmp_path, monkeypatch, rng):
    """Replaying the whole trajectory in one call is what keeps the summing metrics stable."""
    lgbm = get_head("lgbm")
    calls = []

    class CountingSink:
        features = np.zeros((4, 3), dtype=np.float32)

        def record(self, predictions, output="mean"):
            calls.append(len(predictions))

    config = dict(
        learning_rate=0.1,
        num_leaves=7,
        min_child_samples=2,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        lambda_l1=0.0,
        lambda_l2=0.0,
    )
    lgbm.fit(
        "dili",
        rng.random((20, 3)),
        (rng.random(20) > 0.5).astype(float),
        config,
        [CountingSink(), CountingSink()],
        n_iterations=4,
    )
    assert len(calls) == 2
