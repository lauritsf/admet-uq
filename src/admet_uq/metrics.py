"""Metrics: accuracy per compound, UQ per observation.

Accuracy metrics (the TDC leaderboard criteria MAE / AUROC / AP / Spearman, plus
RMSE) score one value per ``compound_id``, so measurement frequency does not bias
them. UQ metrics (NLL / Brier / ECE / CRPS / interval coverage and width) score
every replicate, so a predictive distribution is scored against each measurement
rather than against an average of them.

:func:`aleatoric_floor` reads the same replicates as a measurement-noise scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from admet_uq.data import TASKS, primary_metric

#: Floor on a predictive scale wherever one is divided by, so a row whose spread
#: collapsed gives a large loss or a wide interval rather than a division by zero.
MIN_SCALE = 1e-6

# Spread, relative to a prediction's own magnitude, below which its ordering is
# not an ordering of predictions. Predictions are stored as float32, so a spread
# under that precision does not survive the released artifact.
MIN_RANK_SPREAD = float(np.finfo(np.float32).eps)

# --------------------------------------------------------------------------- #
# Accuracy metrics (operate on already-aggregated per-compound arrays).
# --------------------------------------------------------------------------- #


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def has_ordering(y_pred: np.ndarray, axis: int = -1) -> np.bool_ | np.ndarray:
    """Whether ``y_pred`` spans more than :data:`MIN_RANK_SPREAD` of its magnitude along ``axis``.

    Copies of one repeated value differ in their last bits where they reach a
    metric by different arithmetic paths, and rank against each other.
    """
    values = np.asarray(y_pred, dtype=float)
    return np.ptp(values, axis=axis) > MIN_RANK_SPREAD * np.max(np.abs(values), axis=axis)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation; NaN if the prediction carries no ordering."""
    if not has_ordering(y_pred):
        return float("nan")
    return float(spearmanr(y_true, y_pred)[0])


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC-AUC; NaN if only one class is present."""
    return float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")


def ap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average precision (AUPRC); NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_pred))


def aggregate_by_compound(
    compound_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, task_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse replicate rows to one ``(label, prediction)`` per compound.

    Predictions are averaged. Regression labels are averaged; binary labels take
    the majority (compounds whose replicate labels tie at 0.5 are dropped).
    """
    frame = pd.DataFrame({"cid": np.asarray(compound_id), "y": y_true, "pred": y_pred})
    grouped = frame.groupby("cid", sort=False)
    pred_per_compound = grouped["pred"].mean()
    if task_type == "regression":
        return grouped["y"].mean().to_numpy(), pred_per_compound.to_numpy()
    label_mean = grouped["y"].mean()
    keep = label_mean != 0.5  # drop 0-vs-1 ties (a handful of conflict compounds)
    return (label_mean[keep] >= 0.5).astype(float).to_numpy(), pred_per_compound[keep].to_numpy()


def accuracy_metrics(
    task: str, y_true: np.ndarray, y_pred: np.ndarray, compound_id: np.ndarray
) -> dict[str, float]:
    """Per-compound accuracy metrics for ``task`` (incl. the TDC ``primary`` metric)."""
    task_type = TASKS[task]
    label, pred = aggregate_by_compound(compound_id, y_true, y_pred, task_type)
    if task_type == "regression":
        out = {
            "mae": mae(label, pred),
            "rmse": rmse(label, pred),
            "spearman": spearman(label, pred),
        }
    else:
        out = {"auroc": auroc(label, pred), "ap": ap(label, pred)}
    out["primary"] = out[primary_metric(task)[0]]
    out["n_compounds"] = float(len(label))
    return out


# --------------------------------------------------------------------------- #
# UQ metrics (per observation, every replicate row).
# --------------------------------------------------------------------------- #


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Negative log-likelihood under a Gaussian predictive distribution."""
    std = np.maximum(np.asarray(std), MIN_SCALE)
    return float(np.mean(0.5 * np.log(2 * np.pi * std**2) + (y_true - mean) ** 2 / (2 * std**2)))


def lognormal_nll(y_true: np.ndarray, mean_log: np.ndarray, std_log: np.ndarray) -> float:
    """NLL in original units when ``log(y)`` is Gaussian.

    The change of variables carries the Jacobian of the transform,
    ``-log f_Y(y) = -log f_logY(log y) + log y``. That term keeps the value
    comparable with the identity-transform tasks.
    """
    y_true = np.asarray(y_true, dtype=float)
    return gaussian_nll(np.log(y_true), mean_log, std_log) + float(np.mean(np.log(y_true)))


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution (Gneiting 2005)."""
    std = np.maximum(np.asarray(std), MIN_SCALE)
    z_score = (np.asarray(y_true) - mean) / std
    inner = z_score * (2 * norm.cdf(z_score) - 1) + 2 * norm.pdf(z_score) - 1 / np.sqrt(np.pi)
    return float(np.mean(std * inner))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """PICP: fraction of observations inside [lower, upper]."""
    return float(np.mean((np.asarray(y_true) >= lower) & (np.asarray(y_true) <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """MPIW: mean prediction-interval width."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def interval_score_per_observation(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float
) -> np.ndarray:
    """Winkler interval score of a central-``level`` interval, per observation.

    The width, plus a penalty of ``2 / alpha`` times how far outside the interval
    the observation falls. It is proper for the pair of quantiles the interval
    names, so it orders coverage and width together on one scale, and it is in
    the units of the target.
    """
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    alpha = 1.0 - level
    below = np.maximum(lower - y_true, 0.0)
    above = np.maximum(y_true - upper, 0.0)
    return (upper - lower) + (2.0 / alpha) * (below + above)


def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float) -> float:
    """Mean Winkler interval score."""
    return float(np.mean(interval_score_per_observation(y_true, lower, upper, level)))


def bernoulli_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary negative log-likelihood (log loss)."""
    y_prob = np.clip(np.asarray(y_prob), 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score for binary probabilities."""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins, the estimator the literature reports.

    Binned ECE carries a bias that grows with the number of occupied bins, and
    calibrators change the distribution of predicted probabilities, so occupancy
    varies across the axis being compared. A binning-free alternative is smoothed
    ECE (Blasiok & Nakkiran, ICLR 2024, arXiv:2309.12236; ``relplot`` package).
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    error = 0.0
    for bin_idx in range(n_bins):
        mask = bin_of == bin_idx
        if mask.any():
            error += abs(y_prob[mask].mean() - y_true[mask].mean()) * mask.sum() / len(y_true)
    return float(error)


def regression_uq_metrics(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float = 0.9
) -> dict[str, float]:
    """NLL, CRPS, and central-``level`` interval coverage/width from (mean, std).

    ``mean`` and ``std`` must be in the same space; on the log-transformed tasks
    that is log space, not original units.
    """
    z_crit = norm.ppf(0.5 + level / 2)
    lower, upper = mean - z_crit * std, mean + z_crit * std
    return {
        "nll": gaussian_nll(y_true, mean, std),
        "crps": crps_gaussian(y_true, mean, std),
        "picp": interval_coverage(y_true, lower, upper),
        "mpiw": interval_width(lower, upper),
    }


def binary_uq_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """NLL, Brier, and ECE from predicted probabilities."""
    return {
        "nll": bernoulli_nll(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "ece": ece(y_true, y_prob),
    }


# --------------------------------------------------------------------------- #
# Measurement noise from replicates (labels only, no predictions).
# --------------------------------------------------------------------------- #

# A pooled standard deviation carries a relative standard error near
# ``1 / sqrt(2 * dof)``, so this floor is 7 percent.
MIN_REPLICATE_DOF = 100


def aleatoric_floor(
    task: str,
    compound_id: np.ndarray,
    y_true: np.ndarray,
    min_dof: int = MIN_REPLICATE_DOF,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Pooled within-compound standard deviation, with the test that licenses it.

    The pooled estimate ``sqrt(sum of squared deviations / dof)`` spends one
    degree of freedom per compound, so pairs contribute to it. It describes the
    task only where the replicated compounds stand in for the rest of it, which
    a two-sample Kolmogorov-Smirnov test between the compound-mean targets of the
    replicated and the singleton compounds checks. Repeat measurements arise from
    merging source databases, which selects on the compound as well as on the
    measurement.

    ``reportable`` requires that test to pass at ``alpha`` and at least
    ``min_dof`` degrees of freedom behind the estimate. It is false, rather than
    an error, for a task whose replicates fail either condition.

    Args:
        task: a regression task; the quantity is a scale in the units of ``y``.
        compound_id: parent InChIKey per observation.
        y_true: target per observation, in original units.
        min_dof: degrees of freedom below which the estimate is not reported.
        alpha: level of the representativeness test.

    Returns:
        ``pooled_sd``, its ``dof``, the compound counts either side of the test,
        the test's ``representative_p``, and ``reportable``.
    """
    if TASKS[task] != "regression":
        raise ValueError(f"{task} is binary; replicate disagreement is not a noise scale")

    frame = pd.DataFrame({"cid": np.asarray(compound_id), "y": np.asarray(y_true, dtype=float)})
    grouped = frame.groupby("cid", sort=False)["y"]
    size, compound_mean = grouped.size(), grouped.mean()
    replicated = size > 1

    dof = int((size - 1).sum())
    empty = {
        "pooled_sd": float("nan"),
        "dof": float(dof),
        "n_replicated": float(replicated.sum()),
        "n_singleton": float((~replicated).sum()),
        "representative_p": float("nan"),
        "reportable": 0.0,
    }
    if dof == 0:
        return empty

    squared_deviations = float(((frame["y"] - frame["cid"].map(compound_mean)) ** 2).sum())
    pooled_sd = float(np.sqrt(squared_deviations / dof))
    p_value = float(ks_2samp(compound_mean[replicated], compound_mean[~replicated]).pvalue)
    return empty | {
        "pooled_sd": pooled_sd,
        "representative_p": p_value,
        "reportable": float(dof >= min_dof and p_value > alpha),
    }
