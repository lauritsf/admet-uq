from __future__ import annotations

import numpy as np
import pytest

from admet_uq.curves import (
    EvalSink,
    leaderboard_curve,
    leaderboard_objective,
    nll_objective,
    pad_to,
)
from admet_uq.data import transform_target
from admet_uq.storage import EvalRows, PredictionWriter, read_predictions

AUROC_TASK = "dili"
AP_TASK = "cyp2c9_substrate_carbonmangels"
MAE_TASK = "caco2_wang"
SPEARMAN_LOG_TASK = "half_life_obach"

ALL_METRICS = (AUROC_TASK, AP_TASK, MAE_TASK, SPEARMAN_LOG_TASK)

RULE = "leaderboard"


def make_rows(compound_id, y_true, *, stage="inner", inner_fold=0, eval_set="val") -> EvalRows:
    """An alignment entry over explicit compounds and labels."""
    compound_id = np.asarray(compound_id)
    return EvalRows(
        stage,
        inner_fold,
        eval_set,
        np.arange(len(compound_id), dtype=np.int64),
        compound_id,
        np.asarray(y_true, dtype=np.float64),
    )


def make_case(task: str, rng: np.random.Generator, n_rows: int = 24, n_iterations: int = 7):
    """A trajectory and the rows it is scored against, in the head's output space."""
    # Two compounds carry a replicate, so the per-compound aggregation is exercised.
    compound_id = np.array([f"C{i // 2:03d}" if i < 4 else f"C{i:03d}" for i in range(n_rows)])
    if task in (AUROC_TASK, AP_TASK):
        y_true = (rng.random(n_rows) > 0.4).astype(np.float64)
        trajectory = rng.random((n_iterations, n_rows))
    else:
        y_true = (
            rng.lognormal(0.0, 1.0, n_rows)
            if task == SPEARMAN_LOG_TASK
            else rng.normal(0.0, 1.0, n_rows)
        )
        trajectory = rng.normal(0.0, 1.0, (n_iterations, n_rows))
    return make_rows(compound_id, y_true), trajectory


def record_in_chunks(sink: EvalSink, trajectory: np.ndarray, sizes) -> None:
    """Feed a trajectory to a sink in runs of ``sizes`` iterations."""
    start = 0
    for size in sizes:
        sink.record(trajectory[start : start + size])
        start += size
    assert start == len(trajectory)


@pytest.mark.parametrize("task", [AUROC_TASK, AP_TASK, MAE_TASK, SPEARMAN_LOG_TASK])
def test_one_chunk_reproduces_the_whole_trajectory_curve(task, rng):
    """Recording a trajectory in one call scores exactly what scoring it whole does."""
    rows, trajectory = make_case(task, rng)
    sink = EvalSink(
        task,
        rows,
        "cfg",
        np.empty((len(rows.row_index), 0)),
        scorers={RULE: leaderboard_objective(task, rows).score},
    )
    sink.record(trajectory)
    expected = leaderboard_curve(task, trajectory, rows.y_true, rows.compound_id)
    np.testing.assert_array_equal(sink.curve(RULE), expected)


@pytest.mark.parametrize("task", ALL_METRICS)
def test_every_metric_is_exact_under_any_chunking(task, rng):
    """A curve is scored once, over the whole trajectory, however it arrived.

    The summing metrics reach a different sum when added in a different order, so
    this holds because the sink retains what it is handed and scores at the end,
    not because the metrics are chunk-invariant. Only the rank metrics are that.
    """
    rows, trajectory = make_case(task, rng)
    expected = leaderboard_curve(task, trajectory, rows.y_true, rows.compound_id)
    for sizes in ([1] * 7, [3, 4], [2, 2, 3]):
        sink = EvalSink(
            task,
            rows,
            "cfg",
            np.empty((len(rows.row_index), 0)),
            scorers={RULE: leaderboard_objective(task, rows).score},
        )
        record_in_chunks(sink, trajectory, sizes)
        np.testing.assert_array_equal(sink.curve(RULE), expected)


def test_binary_ties_are_dropped(rng):
    """A compound whose replicates split evenly has no majority label and leaves the metric."""
    compound_id = np.array(["A", "A", "B", "B", "C", "D"])
    rows = make_rows(compound_id, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    trajectory = rng.random((4, 6))

    sink = EvalSink(
        AUROC_TASK,
        rows,
        "cfg",
        np.empty((6, 0)),
        scorers={RULE: leaderboard_objective(AUROC_TASK, rows).score},
    )
    sink.record(trajectory)
    # A ties at 0.5 and goes; B, C and D remain.
    kept = make_rows(np.array(["B", "B", "C", "D"]), [1.0, 1.0, 0.0, 1.0])
    expected = leaderboard_curve(AUROC_TASK, trajectory[:, 2:], kept.y_true, kept.compound_id)
    np.testing.assert_array_equal(sink.curve(RULE), expected)


def test_an_iteration_of_one_repeated_value_scores_nan(rng):
    """Spearman is undefined at an iteration predicting one value, and defined either side."""
    rows, trajectory = make_case(SPEARMAN_LOG_TASK, rng)
    trajectory[2] = np.full(trajectory.shape[1], 0.4)
    trajectory[2, ::3] = np.nextafter(0.4, np.inf)

    curve = leaderboard_curve(SPEARMAN_LOG_TASK, trajectory, rows.y_true, rows.compound_id)
    assert np.isnan(curve[2])
    assert np.isfinite(np.delete(curve, 2)).all()


def test_single_class_fold_scores_nan(rng):
    """AUROC is undefined when every compound carries the same label."""
    rows = make_rows(np.array(["A", "B", "C"]), [1.0, 1.0, 1.0])
    sink = EvalSink(
        AUROC_TASK,
        rows,
        "cfg",
        np.empty((3, 0)),
        scorers={RULE: leaderboard_objective(AUROC_TASK, rows).score},
    )
    sink.record(rng.random((5, 3)))
    assert np.isnan(sink.curve(RULE)).all()
    assert len(sink.curve(RULE)) == 5


def test_log_task_is_scored_in_original_units(rng):
    """The sink inverts the training-space log transform before scoring and storing."""
    rows, trajectory = make_case(SPEARMAN_LOG_TASK, rng)
    sink = EvalSink(
        SPEARMAN_LOG_TASK,
        rows,
        "cfg",
        np.empty((len(rows.row_index), 0)),
        scorers={RULE: leaderboard_objective(SPEARMAN_LOG_TASK, rows).score},
    )
    sink.record(trajectory)
    expected = leaderboard_curve(SPEARMAN_LOG_TASK, trajectory, rows.y_true, rows.compound_id)
    np.testing.assert_array_equal(sink.curve(RULE), expected)


@pytest.mark.parametrize("keep", [1, 3, 7, 9])
@pytest.mark.parametrize("sizes", [[7], [1] * 7, [2, 2, 3], [5, 2]])
def test_kept_is_the_clamped_iteration(keep, sizes, rng):
    """``kept`` is the requested iteration, or the last one recorded when the fit ends first."""
    rows, trajectory = make_case(MAE_TASK, rng)
    sink = EvalSink(
        MAE_TASK, rows, "cfg", np.empty((len(rows.row_index), 0)), keep_iterations=[keep]
    )
    record_in_chunks(sink, trajectory, sizes)
    expected = trajectory[min(keep, len(trajectory)) - 1].astype(np.float32)
    np.testing.assert_array_equal(sink.kept[keep], expected)


def test_several_iterations_are_kept_from_one_fit(rng):
    """Two rules landing on one config deploy it at their own iterations."""
    rows, trajectory = make_case(MAE_TASK, rng)
    sink = EvalSink(
        MAE_TASK, rows, "cfg", np.empty((len(rows.row_index), 0)), keep_iterations=[2, 5]
    )
    record_in_chunks(sink, trajectory, [3, 4])
    assert sorted(sink.kept) == [2, 5]
    for wanted in (2, 5):
        np.testing.assert_array_equal(sink.kept[wanted], trajectory[wanted - 1].astype(np.float32))


def test_nothing_is_kept_without_a_keep_iteration(rng):
    """A sink that is only asked to store retains no vector."""
    rows, trajectory = make_case(MAE_TASK, rng)
    sink = EvalSink(MAE_TASK, rows, "cfg", np.empty((len(rows.row_index), 0)))
    sink.record(trajectory)
    assert sink.kept == {}


def test_a_scale_output_is_stored_in_the_modelling_space(tmp_path, rng):
    """A spread carries no location, so the log task's inverse transform skips it."""
    rows, trajectory = make_case(SPEARMAN_LOG_TASK, rng)
    spread = np.abs(trajectory) + 0.1
    path = tmp_path / "predictions.parquet"
    with PredictionWriter(path) as writer:
        sink = EvalSink(
            SPEARMAN_LOG_TASK,
            rows,
            "cfg",
            np.empty((len(rows.row_index), 0)),
            writer=writer,
            scale_outputs=["std"],
        )
        sink.record(trajectory)
        sink.record(spread, output="std")

    stored = read_predictions(path).sort_values(["output", "iteration"])
    means = np.stack(stored[stored["output"] == "mean"]["pred"].to_numpy())
    scales = np.stack(stored[stored["output"] == "std"]["pred"].to_numpy())
    np.testing.assert_allclose(means, np.exp(trajectory), rtol=1e-6)
    np.testing.assert_allclose(scales, spread, rtol=1e-6)


def test_the_regression_likelihood_reads_the_scale_where_it_lives(rng):
    """The mean is mapped back into the scale's space before the two are combined."""
    rows, trajectory = make_case(SPEARMAN_LOG_TASK, rng)
    spread = np.abs(trajectory) + 0.1
    sink = EvalSink(
        SPEARMAN_LOG_TASK,
        rows,
        "cfg",
        np.empty((len(rows.row_index), 0)),
        scale_outputs=["std"],
        scorers={"nll": nll_objective(SPEARMAN_LOG_TASK, rows).score},
    )
    sink.record(trajectory)
    sink.record(spread, output="std")

    target = transform_target(SPEARMAN_LOG_TASK, rows.y_true)
    expected = np.mean(
        0.5 * np.log(2 * np.pi * spread**2) + (target - trajectory) ** 2 / (2 * spread**2),
        axis=1,
    )
    np.testing.assert_allclose(sink.curve("nll"), expected, rtol=0, atol=1e-10)


def test_binary_nll_scores_the_probability(rng):
    """A Bernoulli predictive is one number, so the rule reads the mean alone."""
    rows, trajectory = make_case(AUROC_TASK, rng)
    sink = EvalSink(
        AUROC_TASK,
        rows,
        "cfg",
        np.empty((len(rows.row_index), 0)),
        scorers={"nll": nll_objective(AUROC_TASK, rows).score},
    )
    sink.record(trajectory)
    expected = -np.mean(
        rows.y_true * np.log(trajectory) + (1 - rows.y_true) * np.log(1 - trajectory), axis=1
    )
    np.testing.assert_allclose(sink.curve("nll"), expected, rtol=0, atol=1e-12)


def test_a_sink_scores_every_rule_it_was_given(rng):
    """One sweep feeds every rule the head named, off the same recorded predictions."""
    rows, trajectory = make_case(AUROC_TASK, rng)
    sink = EvalSink(
        AUROC_TASK,
        rows,
        "cfg",
        np.empty((len(rows.row_index), 0)),
        scorers={
            RULE: leaderboard_objective(AUROC_TASK, rows).score,
            "nll": nll_objective(AUROC_TASK, rows).score,
        },
    )
    sink.record(trajectory)
    leaderboard = sink.curve(RULE)
    assert len(leaderboard) == len(trajectory)
    np.testing.assert_array_equal(
        leaderboard, leaderboard_curve(AUROC_TASK, trajectory, rows.y_true, rows.compound_id)
    )
    assert np.all(sink.curve("nll") > 0)


@pytest.mark.parametrize("sizes", [[1] * 7, [3, 4], [2, 2, 3]])
def test_chunked_and_whole_recording_store_the_same_rows(sizes, tmp_path, rng):
    """Iteration numbering continues across calls, so storage cannot tell the two apart."""
    rows, trajectory = make_case(MAE_TASK, rng)
    features = np.empty((len(rows.row_index), 0))

    paths = {}
    for name, chunking in (("whole", [len(trajectory)]), ("chunked", sizes)):
        paths[name] = tmp_path / f"{name}.parquet"
        with PredictionWriter(paths[name]) as writer:
            sink = EvalSink(MAE_TASK, rows, "cfg", features, writer=writer)
            record_in_chunks(sink, trajectory, chunking)

    whole = read_predictions(paths["whole"]).sort_values("iteration").reset_index(drop=True)
    chunked = read_predictions(paths["chunked"]).sort_values("iteration").reset_index(drop=True)
    np.testing.assert_array_equal(whole["iteration"], np.arange(1, len(trajectory) + 1))
    np.testing.assert_array_equal(whole["iteration"], chunked["iteration"])
    np.testing.assert_array_equal(np.stack(whole["pred"]), np.stack(chunked["pred"]))


def test_pad_to_repeats_the_last_value():
    """A fit that ends early leaves the curve constant, which is what the model does."""
    np.testing.assert_array_equal(pad_to(np.array([1.0, 2.0]), 5), [1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(pad_to(np.array([1.0, 2.0]), 2), [1.0, 2.0])
