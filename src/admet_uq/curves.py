"""Selection objectives and sinks for per-iteration predictions.

An :class:`EvalSink` scores trajectories and streams predictions to storage.
Location outputs are converted to original target units; declared scale outputs
remain in model space.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from admet_uq.data import TASKS, inverse_target, primary_metric, transform_target
from admet_uq.metrics import MIN_SCALE, has_ordering
from admet_uq.storage import EvalRows, PredictionBlock, PredictionWriter


class _CompoundAggregator:
    """Collapses replicate rows to one value per compound, for a whole trajectory.

    Rows are sorted by ``compound_id`` once, then every iteration is reduced in
    a single ``np.add.reduceat`` over the ``(n_iterations, n_rows)`` matrix.

    Binary tasks take the majority replicate label and drop compounds whose
    replicates tie at 0.5, matching ``metrics.aggregate_by_compound``.
    """

    def __init__(self, compound_id: np.ndarray, y_true: np.ndarray, task_type: str):
        order = np.argsort(np.asarray(compound_id), kind="stable")
        sorted_ids = np.asarray(compound_id)[order]
        starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
        counts = np.diff(np.r_[starts, len(sorted_ids)])

        label = np.add.reduceat(np.asarray(y_true, dtype=np.float64)[order], starts) / counts
        keep = np.ones(len(label), dtype=bool)
        if task_type == "binary":
            keep = label != 0.5
            label = (label[keep] >= 0.5).astype(np.float64)

        self._order, self._starts, self._counts, self._keep = order, starts, counts, keep
        self.label = label

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        aggregated = np.add.reduceat(trajectory[:, self._order], self._starts, axis=1)
        return (aggregated / self._counts)[:, self._keep]


def _metric_curve(metric: str, label: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Evaluate ``metric`` for every row of a ``(n_iterations, n_compounds)`` matrix."""
    if metric == "mae":
        return np.mean(np.abs(predictions - label), axis=1)
    if metric == "rmse":
        return np.sqrt(np.mean((predictions - label) ** 2, axis=1))
    if metric == "spearman":
        # Spearman is Pearson on ranks, done in matrix form over the trajectory.
        rank_pred = rankdata(predictions, axis=1)
        rank_true = rankdata(label)
        centred_pred = rank_pred - rank_pred.mean(axis=1, keepdims=True)
        centred_true = rank_true - rank_true.mean()
        denominator = np.sqrt((centred_pred**2).sum(axis=1) * (centred_true**2).sum())
        return np.divide(
            centred_pred @ centred_true,
            denominator,
            out=np.full(len(predictions), np.nan),
            where=(denominator > 0) & has_ordering(predictions, axis=1),
        )
    if len(np.unique(label)) < 2:
        # A cluster-grouped inner fold can be single-class on a small, imbalanced
        # task, where AUROC/AP are undefined.
        return np.full(len(predictions), np.nan)
    return _auroc_curve(label, predictions) if metric == "auroc" else _ap_curve(label, predictions)


def _auroc_curve(label: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """AUROC for every row, via the Mann-Whitney U identity.

    ``rankdata`` assigns midranks to ties, which is the tie handling of the
    trapezoidal ROC that ``roc_auc_score`` integrates.
    """
    positive = label > 0.5
    n_pos = int(positive.sum())
    n_neg = len(label) - n_pos
    ranks = rankdata(predictions, axis=1)
    return (ranks[:, positive].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _ap_curve(label: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Average precision for every row, matching ``average_precision_score``.

    sklearn sums ``(recall_i - recall_{i-1}) * precision_i`` over distinct score
    thresholds, so a run of tied scores contributes a single step taken at the
    end of the run. ``recall`` is non-decreasing, so a running maximum over the
    recall values at run-ends, shifted by one, gives each run's predecessor.
    """
    positive = (label > 0.5).astype(np.float64)
    n_pos = positive.sum()
    order = np.argsort(-predictions, axis=1, kind="stable")
    hits = positive[order]
    true_positives = np.cumsum(hits, axis=1)
    precision = true_positives / np.arange(1, predictions.shape[1] + 1)
    recall = true_positives / n_pos

    # A position ends a tied run when the next score differs, as does the last.
    sorted_scores = np.take_along_axis(predictions, order, axis=1)
    run_end = np.ones(predictions.shape, dtype=bool)
    run_end[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]

    previous = np.zeros_like(recall)
    previous[:, 1:] = np.maximum.accumulate(np.where(run_end, recall, 0.0), axis=1)[:, :-1]
    return np.sum(np.where(run_end, (recall - previous) * precision, 0.0), axis=1)


def leaderboard_curve(
    task: str, trajectory: np.ndarray, y_true: np.ndarray, compound_id: np.ndarray
) -> np.ndarray:
    """Per-iteration leaderboard metric for one fit, in the task's original units.

    ``trajectory`` is ``(n_iterations, n_rows)``: probabilities for binary tasks,
    training-space values for regression.
    """
    if TASKS[task] == "regression":
        trajectory = inverse_target(task, trajectory)
    aggregator = _CompoundAggregator(compound_id, y_true, TASKS[task])
    return _metric_curve(primary_metric(task)[0], aggregator.label, aggregator(trajectory))


@dataclass(frozen=True)
class Objective:
    """What a head is selected on: a named, directed score over a trajectory.

    ``score`` takes every output of one fit, keyed by name and each
    ``(n_iterations, n_rows)``, and returns one value per iteration. Locations
    arrive in the task's original units and scales in the modelling space.
    """

    name: str
    direction: str
    score: Callable[[Mapping[str, np.ndarray]], np.ndarray]


def leaderboard_objective(task: str, rows: EvalRows, output: str = "mean") -> Objective:
    """The task's leaderboard metric on ``output``, aggregated per compound.

    Measurement frequency does not bias the score, because replicates of a
    compound are averaged before the metric is taken.
    """
    aggregator = _CompoundAggregator(rows.compound_id, rows.y_true, TASKS[task])
    name, direction = primary_metric(task)

    def score(values: Mapping[str, np.ndarray]) -> np.ndarray:
        return _metric_curve(name, aggregator.label, aggregator(values[output]))

    return Objective(name, direction, score)


def pinball_objective(rows: EvalRows, quantiles: dict[str, float]) -> Objective:
    """Pinball loss averaged over the fitted quantiles, scored per observation.

    ``quantiles`` maps an output name to the quantile it estimates. Summing the
    two losses of a central pair gives that interval's score up to a constant
    factor, so averaging over the outputs scores the interval and the median
    together.
    """
    y_true = np.asarray(rows.y_true, dtype=np.float64)

    def one(alpha: float, values: np.ndarray) -> np.ndarray:
        residual = y_true - values
        return np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual), axis=1)

    def score(values: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.mean([one(alpha, values[name]) for name, alpha in quantiles.items()], axis=0)

    return Objective("pinball", "minimize", score)


def nll_objective(
    task: str, rows: EvalRows, mean_output: str = "mean", scale_output: str = "std"
) -> Objective:
    """Negative log-likelihood of the predictive distribution, per observation.

    Binary tasks score the Bernoulli likelihood of the predicted probability,
    which the leaderboard's ranking metrics are blind to. Regression tasks score
    a Gaussian in the modelling space, which is where the head's scale lives and
    where the residuals of a log-transformed PK task are symmetric; the change of
    variables back to original units contributes a term that is constant across
    configs and iterations, so it is left out.
    """
    y_true = np.asarray(rows.y_true, dtype=np.float64)
    if TASKS[task] == "binary":

        def score(values: Mapping[str, np.ndarray]) -> np.ndarray:
            probability = np.clip(values[mean_output], 1e-7, 1 - 1e-7)
            per_row = -(y_true * np.log(probability) + (1 - y_true) * np.log(1 - probability))
            return np.mean(per_row, axis=1)

        return Objective("nll", "minimize", score)

    target = transform_target(task, y_true)

    def score(values: Mapping[str, np.ndarray]) -> np.ndarray:
        mean = transform_target(task, values[mean_output])
        scale = np.maximum(values[scale_output], MIN_SCALE)
        per_row = 0.5 * np.log(2 * np.pi * scale**2) + (target - mean) ** 2 / (2 * scale**2)
        return np.mean(per_row, axis=1)

    return Objective("nll", "minimize", score)


def pad_to(values: np.ndarray, length: int) -> np.ndarray:
    """Extend along the first axis to ``length`` by repeating the last entry.

    A fit can end below ``length`` when training stops adding to the model. The
    model is unchanged past that point, so its last curve value or last
    prediction vector continues the trajectory, and the mean over inner folds is
    taken over equal-length curves.
    """
    if len(values) == length:
        return values
    return np.concatenate([values, np.repeat(values[-1:], length - len(values), axis=0)])


class EvalSink:
    """One evaluation set of one fit: the rows to predict, and where predictions go.

    ``features`` is what the head predicts; every call to :meth:`record` passes
    the predictions for a run of consecutive iterations of one output,
    continuing from the last call for that output. Recording the whole
    trajectory in one call and recording it an iteration at a time both give the
    same rows in storage and the same curves.

    Curves are scored only when ``scorers`` is given, since the refit's
    calibration and test predictions are stored rather than selected on. A
    scoring sink retains the predictions it is handed until :meth:`curve` asks
    for them, because a rule reads every output of an iteration together.
    """

    def __init__(
        self,
        task: str,
        rows: EvalRows,
        config_id: str,
        features: np.ndarray,
        *,
        writer: PredictionWriter | None = None,
        scorers: Mapping[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] | None = None,
        scale_outputs: Iterable[str] = (),
        keep_iterations: Iterable[int] = (),
        keep_output: str = "mean",
    ):
        """Target one evaluation set.

        Args:
            task: TDC ADMET task name.
            rows: what the predictions align to, and the labels to score against.
            config_id: the config being fitted, carried into storage.
            features: the rows to predict, ``(n_rows, n_features)``.
            writer: where blocks are streamed; ``None`` stores nothing.
            scorers: one score function per selection rule; ``None`` scores
                nothing.
            scale_outputs: outputs holding a spread, which the target transform
                is not inverted on.
            keep_iterations: iterations to retain in :attr:`kept`, each clamped
                to the last one recorded. One fit can be deployed at several
                iterations, one per selection rule that chose its config.
            keep_output: which output :attr:`kept` holds.
        """
        self.features = features
        self.kept: dict[int, np.ndarray] = {}
        self._task = task
        self._rows = rows
        self._config_id = config_id
        self._writer = writer
        self._scorers = dict(scorers or {})
        self._scale_outputs = frozenset(scale_outputs)
        self._keep_iterations = tuple(keep_iterations)
        self._keep_output = keep_output
        self._retained: dict[str, list[np.ndarray]] = {}
        self._recorded: dict[str, int] = {}

    def record(self, predictions: np.ndarray, output: str = "mean") -> None:
        """Take ``(n_iterations, n_rows)`` predictions for the iterations after the last call."""
        values = np.ascontiguousarray(predictions, dtype=np.float64)
        if TASKS[self._task] == "regression" and output not in self._scale_outputs:
            values = inverse_target(self._task, values)
        recorded = self._recorded.get(output, 0)
        if self._scorers:
            self._retained.setdefault(output, []).append(values)
        block = PredictionBlock(
            self._rows.stage,
            self._config_id,
            self._rows.inner_fold,
            self._rows.eval_set,
            values,
            first_iteration=recorded + 1,
            output=output,
        )
        if output == self._keep_output:
            for wanted in self._keep_iterations:
                index = min(wanted, recorded + len(values)) - 1 - recorded
                if index >= 0:
                    self.kept[wanted] = block.preds[index].copy()
        if self._writer is not None:
            self._writer.write(block)
        self._recorded[output] = recorded + len(values)

    def curve(self, rule: str) -> np.ndarray:
        """The score of ``rule`` at every iteration recorded.

        Outputs of one fit can end at different iterations, since a model per
        output stops when that output stops improving. A short one is continued
        by its last prediction, which is what it would predict there.
        """
        values = {name: np.concatenate(chunks) for name, chunks in self._retained.items()}
        longest = max(len(trajectory) for trajectory in values.values())
        return self._scorers[rule](
            {name: pad_to(trajectory, longest) for name, trajectory in values.items()}
        )


#: Separates the arm from the output in the name a group fit records under.
ARM_SEPARATOR = ":"


class GroupSink:
    """One evaluation set of a fit serving several arms at once.

    A group fit trains one network and reads it through every arm in the group,
    so it produces one trajectory per ``arm:output`` pair over a single set of
    rows. This holds those rows once and hands each value to the sink of the arm
    that produced it, which is what keeps every arm's own cell unchanged.
    """

    def __init__(self, sinks: Mapping[str, EvalSink]):
        """Route to one sink per arm, all holding the same rows."""
        self.sinks = dict(sinks)
        features = {id(sink.features) for sink in self.sinks.values()}
        if len(features) != 1:
            raise ValueError("a group's sinks must predict one shared set of rows")
        self.features = next(iter(self.sinks.values())).features

    def record(self, predictions: np.ndarray, output: str) -> None:
        """Pass one ``arm:output`` trajectory to that arm's sink under its own name."""
        arm, _, name = output.partition(ARM_SEPARATOR)
        self.sinks[arm].record(predictions, output=name)
