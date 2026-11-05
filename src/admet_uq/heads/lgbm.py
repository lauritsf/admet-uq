"""LightGBM head.

The presampled configs search learning rate, head shape and regularisation; the
objective is locked per task and aligned to the TDC leaderboard criterion
(MAE tasks -> L1; Spearman-scored regression -> L2; binary -> logloss, scored on
AUC / average-precision).

Every fit runs the full ``n_iterations`` budget; the driver tunes the iteration
on the inner CV rather than stopping in-loop.

A fitted booster holds every tree, so the prediction after ``k`` rounds is a
running sum over the first ``k`` of them, recoverable for any feature matrix
once training is done. The trajectory therefore reaches the evaluation sinks in
one replay at the end, where a head carrying only its current state has to
record as it goes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np

from admet_uq.curves import EvalSink, Objective, leaderboard_objective
from admet_uq.data import TASKS, primary_metric
from admet_uq.heads import DEFAULT_SELECTION, DETERMINISTIC, FitResult, Head
from admet_uq.storage import EvalRows

HEAD = "lgbm"
UQ = DETERMINISTIC
DEFAULT_ITERATIONS = 2000
OUTPUTS = ("mean",)
SCALE_OUTPUTS = ()
POINT_OUTPUT = "mean"
CONFIG_LIST = "lgbm"
TASK_TYPES = ("binary", "regression")


def objectives(task: str, rows: EvalRows) -> dict[str, Objective]:
    """Select on the leaderboard metric of the task."""
    return {DEFAULT_SELECTION: leaderboard_objective(task, rows, POINT_OUTPUT)}


def _objective_metric(task: str) -> tuple[str, str]:
    """Locked (objective, validation metric) for a task, leaderboard-aligned."""
    metric = primary_metric(task)[0]
    if TASKS[task] == "binary":
        return "binary", ("auc" if metric == "auroc" else "average_precision")
    if metric == "mae":
        return "regression_l1", "l1"
    return "regression", "l2"  # Spearman-scored regression


def fit(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    eval_sinks: Sequence[EvalSink],
    *,
    seed: int = 0,
    n_jobs: int = 0,
    n_iterations: int = DEFAULT_ITERATIONS,
    artifacts_dir: Path | None = None,
    checkpoint_at: int | None = None,
) -> FitResult:
    """Train to the full round budget, then replay the trajectory onto each sink.

    Each sink is recorded in a single call, which is what keeps the summing
    metrics identical to scoring the trajectory whole.
    """
    model = train(
        task, x_train, y_train, config, seed=seed, n_jobs=n_jobs, n_iterations=n_iterations
    )
    for sink in eval_sinks:
        sink.record(predict_trajectory(task, model, sink.features))
    return FitResult(model.num_trees())


def train(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    *,
    seed: int = 0,
    n_jobs: int = 0,
    n_iterations: int = DEFAULT_ITERATIONS,
    params_override: dict | None = None,
) -> lgb.Booster:
    """Train to the full round budget.

    ``params_override`` replaces entries of the parameter dict, which is how a
    head fitting a different loss on the same search space names it.
    """
    loss, metric = _objective_metric(task)
    params = {
        "objective": loss,
        "metric": metric,
        "verbosity": -1,
        "seed": seed,
        "num_threads": n_jobs,
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "min_child_samples": int(config["min_child_samples"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": 1,
        "lambda_l1": float(config["lambda_l1"]),
        "lambda_l2": float(config["lambda_l2"]),
    }
    params |= params_override or {}
    return lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train),
        num_boost_round=n_iterations,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def predict_trajectory(task: str, model: lgb.Booster, features: np.ndarray) -> np.ndarray:
    """Predictions after every boosting iteration: ``(n_trees, n_rows)``.

    Boosting is additive in raw score, so the trajectory is the running sum of
    the per-tree raw contributions. LightGBM folds the ``boost_from_average``
    init score into the first slice and does not repeat it, so a plain ``cumsum``
    reproduces ``predict(num_iteration=k, raw_score=True)``. The inverse link is
    applied afterwards, once, over the whole matrix.
    """
    n_trees = model.num_trees()
    raw = np.cumsum(
        np.stack(
            [
                model.predict(features, start_iteration=i, num_iteration=1, raw_score=True)
                for i in range(n_trees)
            ]
        ),
        axis=0,
    )
    if TASKS[task] == "binary":
        # Same logistic link LightGBM applies internally for objective="binary".
        return 1.0 / (1.0 + np.exp(-raw))
    return raw


SPEC = Head(
    HEAD=HEAD,
    UQ=UQ,
    DEFAULT_ITERATIONS=DEFAULT_ITERATIONS,
    OUTPUTS=OUTPUTS,
    SCALE_OUTPUTS=SCALE_OUTPUTS,
    POINT_OUTPUT=POINT_OUTPUT,
    CONFIG_LIST=CONFIG_LIST,
    TASK_TYPES=TASK_TYPES,
    fit=fit,
    objectives=objectives,
)
