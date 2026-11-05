"""Collect per-cell artifacts into the two benchmark-level results tables.

A cell is one ``(task, head, outer fold)``. It stores uncalibrated predictions at
every iteration, for both the test fold and the calibration slice, so every
calibrator is a pure function of what is already on disk and the whole panel is
built here at read time from the deployed iteration alone.

Both tables are keyed by a method, the quadruple
``(head, uq, selection, calibration)``:

``observations``
    The instance-level unit. One row per ``(cell, method, evaluation
    observation)``: ``compound_id``, ``y_true``, the predictive output and the
    per-observation losses.

``folds``
    The dataset-level unit. One row per ``(cell, method, metric)``.

:func:`null_tables` fills the same two tables for a predictor that ignores the
structure. It has no cell, so it is fitted on the curated labels of the rows the
cells fitted on and scored on the rows they were scored on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from admet_uq.calibrate import (
    CLASSIFICATION_CALIBRATORS,
    DISTRIBUTIONAL_CALIBRATORS,
    REGRESSION_CALIBRATORS,
    Calibrated,
    calibrate_classification,
    fit_gaussian_scale,
    fit_noise_floor,
    fit_scale_factor,
)
from admet_uq.data import TASKS, load_task, primary_metric, target_transform
from admet_uq.heads import DEFAULT_SELECTION, arm_directory
from admet_uq.heads import DETERMINISTIC as DEFAULT_UQ
from admet_uq.metrics import (
    accuracy_metrics,
    binary_uq_metrics,
    gaussian_nll,
    interval_score,
    interval_score_per_observation,
    lognormal_nll,
)
from admet_uq.splits import OuterFold, read_splits
from admet_uq.storage import EVAL_ROWS_FILE, METADATA_FILE, PREDICTIONS_FILE, read_predictions

# Columns identifying a method. Every statistic groups by these.
METHOD_KEYS = ("head", "uq", "selection", "calibration")

# Nominal central coverage for every interval and label-set calibrator.
DEFAULT_LEVEL = 0.90

OBSERVATIONS_FILE = "observations.parquet"
FOLDS_FILE = "folds.parquet"

# Per-observation loss columns. A calibrator fills the ones that apply to it and
# leaves the rest NaN, so the table stays one shape across calibrator families.
LOSS_COLUMNS = (
    "nll",
    "brier",
    "abs_error",
    "sq_error",
    "covered",
    "width",
    "interval_score",
    "set_size",
)


def iter_cells(results_root: str | Path) -> Iterator[Path]:
    """Yield each cell directory holding a ``cell.json``, sorted."""
    return iter(sorted(Path(results_root).glob(f"*/*/outer*/{METADATA_FILE}"), key=str))


def read_cell(metadata_path: str | Path) -> dict:
    """Read one ``cell.json``, adding the identity implied by its path.

    A cell written before the ``uq`` field existed is a deterministic one, whose
    arm directory is its head name. One written before a head could name several
    selection rules carries its single rule in the flat fields, which is what
    :func:`cell_selections` reads it back out of.
    """
    metadata_path = Path(metadata_path)
    cell_dir = metadata_path.parent
    meta = json.loads(metadata_path.read_text())
    meta["cell_dir"] = str(cell_dir)
    meta.setdefault("uq", DEFAULT_UQ)
    meta.setdefault("scale_outputs", [])
    # The path is authoritative for identity; cell.json repeats it, so disagreeing
    # values mean a cell was written into the wrong directory.
    for field, found, value in (
        ("task", meta.get("task"), cell_dir.parent.parent.name),
        ("head", arm_directory(meta["head"], meta["uq"]), cell_dir.parent.name),
        ("outer", meta.get("outer"), int(cell_dir.name.removeprefix("outer"))),
    ):
        if found != value:
            raise ValueError(
                f"{metadata_path}: cell.json {field}={found!r} contradicts path {value!r}"
            )
    return meta


def cell_selections(meta: dict) -> list[dict]:
    """Every selection rule the cell was refitted under, in the order it ran them.

    Each entry names its rule, the config it chose and the iteration deployed
    under it. A cell predating the ``selections`` block ran one rule, held in the
    flat fields.
    """
    if "selections" in meta:
        return list(meta["selections"].values())
    return [
        {
            "selection": meta.get("selection", DEFAULT_SELECTION),
            "selected_config_id": meta["selected_config_id"],
            "deployed_iteration": meta["deployed_iteration"],
        }
    ]


def read_final(meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A cell's final-stage prediction blocks and its evaluation rows.

    Every selection rule, evaluation set and output of a cell reads a slice of
    these same rows. The unit a reader pulls off disk is the row group holding
    them, so one pass serves every slice where a read per slice pays for that
    group again each time.
    """
    cell_dir = Path(meta["cell_dir"])
    return (
        read_predictions(cell_dir / PREDICTIONS_FILE, stage="final"),
        pd.read_parquet(cell_dir / EVAL_ROWS_FILE),
    )


def deployed_predictions(
    meta: dict,
    selection: dict,
    eval_set: str,
    output: str | None = None,
    *,
    final: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """One evaluation set's rows and predictions at a rule's deployed iteration.

    ``eval_set`` is ``"calib"`` or ``"test"``; ``output`` names which value of a
    multi-output arm to read, defaulting to the cell's point prediction. Rows come
    back in stored order, so ``row_index`` still points into the task-level
    feature arrays. ``final`` is what :func:`read_final` returns, for a caller
    reading several slices of one cell.

    The block is picked by config as well as by iteration, since two rules
    choosing different configs give a cell two refits.

    An output that ended before the deployed iteration is read at the last one it
    holds, which is the prediction it makes from there on. The quantile arm fits
    a booster per output and they stop at different rounds, so reading them all
    at one iteration would either fail or pull every output back to the shortest.
    An iteration past the longest output of the fit is reached by none of them,
    and raises rather than reading the end of the trajectory.
    """
    cell_dir = Path(meta["cell_dir"])
    predictions, rows = read_final(meta) if final is None else final
    rows = rows[(rows["stage"] == "final") & (rows["eval_set"] == eval_set)]

    refit = (
        (predictions["stage"] == "final")
        & (predictions["eval_set"] == eval_set)
        & (predictions["config_id"] == selection["selected_config_id"])
    )
    selected = refit
    # Schema 1 stores a single unnamed output per row.
    if meta.get("schema_version", 1) >= 2:
        selected = refit & (predictions["output"] == (output or meta["point_output"]))
    candidates = predictions[selected]
    iteration = selection["deployed_iteration"]
    if iteration <= predictions[refit]["iteration"].max():
        iteration = min(iteration, candidates["iteration"].max())
    preds = candidates[candidates["iteration"] == iteration]
    if len(preds) != 1:
        raise ValueError(
            f"{cell_dir}: expected exactly one final/{eval_set} block at iteration "
            f"{iteration}, found {len(preds)}"
        )
    y_pred = np.asarray(preds["pred"].iloc[0], dtype=float)
    if len(y_pred) != len(rows):
        raise ValueError(
            f"{cell_dir}: {eval_set} prediction vector has {len(y_pred)} entries against "
            f"{len(rows)} evaluation rows"
        )
    return pd.DataFrame(
        {
            "row_index": rows["row_index"].to_numpy(),
            "compound_id": rows["compound_id"].to_numpy(),
            "y_true": rows["y_true"].to_numpy(dtype=float),
            "y_pred": y_pred,
        }
    )


def verify_cell(cell_dir: str | Path) -> None:
    """Check that every deployed calibration and test prediction can be read."""
    metadata = read_cell(Path(cell_dir) / METADATA_FILE)
    for selection in cell_selections(metadata):
        for eval_set in ("calib", "test"):
            deployed_predictions(metadata, selection, eval_set)


def _to_model_space(values: np.ndarray, transform: str, what: str) -> np.ndarray:
    """Map original-unit values into the space an interval should be fitted in."""
    if transform != "log":
        return np.asarray(values, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError(f"{what}: log-transformed task has non-positive values")
    return np.log(values)


def _regression_calibrator(
    calibrator: str,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    transform: str,
    level: float,
) -> Calibrated:
    """Fit one regression calibrator in model space and map its interval back.

    The output is in original units; ``Calibrated.scale`` carries the fitted
    Gaussian sigma, which stays in model space.
    """
    calib_pred = _to_model_space(calib["y_pred"], transform, "calibration prediction")
    calib_true = _to_model_space(calib["y_true"], transform, "calibration label")
    test_pred = _to_model_space(test["y_pred"], transform, "test prediction")
    if calibrator == "gaussian":
        sigma = fit_gaussian_scale(calib_pred, calib_true)
        z_critical = norm.ppf(0.5 + level / 2)
        lower, upper = test_pred - z_critical * sigma, test_pred + z_critical * sigma
        scale = sigma
        pred_std = np.full(len(test_pred), sigma)
    else:
        raise ValueError(f"unknown regression calibrator {calibrator!r}")

    if transform == "log":
        # exp is strictly increasing, so endpoints map without reordering and the
        # interval becomes correctly asymmetric in original units.
        lower, upper = np.exp(lower), np.exp(upper)
    return Calibrated(
        y_pred=test["y_pred"].to_numpy(dtype=float),
        pred_std=pred_std,
        lower=lower,
        upper=upper,
        scale=float(scale),
    )


def _distributional_calibrator(
    calibrator: str,
    calib: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    transform: str,
    level: float,
) -> Calibrated:
    """Report a head's own predictive scale, rescale it, add to it, or replace it.

    ``calib`` and ``test`` hold the point and scale frames of one evaluation set,
    keyed ``point`` / ``scale``. The scale is already in model space, which is
    where it was produced and where the interval is formed.

    All five give an interval; ``none``, ``scaled`` and ``noise_floor`` also give
    a density, and set ``pred_std`` so that the likelihood is scored against it.
    """
    calib_pred = _to_model_space(calib["point"]["y_pred"], transform, "calibration prediction")
    calib_true = _to_model_space(calib["point"]["y_true"], transform, "calibration label")
    calib_scale = calib["scale"]["y_pred"].to_numpy(dtype=float)
    test_pred = _to_model_space(test["point"]["y_pred"], transform, "test prediction")
    test_scale = test["scale"]["y_pred"].to_numpy(dtype=float)
    if calibrator in ("none", "scaled", "noise_floor"):
        if calibrator == "none":
            factor, pred_std = 1.0, test_scale
        elif calibrator == "scaled":
            factor = fit_scale_factor(calib_pred, calib_true, calib_scale)
            pred_std = factor * test_scale
        else:
            factor = fit_noise_floor(calib_pred, calib_true, calib_scale)
            pred_std = np.sqrt(test_scale**2 + factor**2)
        z_critical = norm.ppf(0.5 + level / 2)
        lower, upper = test_pred - z_critical * pred_std, test_pred + z_critical * pred_std
    else:
        raise ValueError(f"unknown distributional calibrator {calibrator!r}")

    if transform == "log":
        lower, upper = np.exp(lower), np.exp(upper)
    return Calibrated(
        y_pred=test["point"]["y_pred"].to_numpy(dtype=float),
        pred_std=pred_std,
        lower=lower,
        upper=upper,
        scale=float(factor),
    )


def _classification_calibrator(
    calibrator: str, calib: pd.DataFrame, test: pd.DataFrame, level: float
) -> Calibrated:
    """Fit one classification calibrator on the calibration slice."""
    return calibrate_classification(
        calibrator,
        calib["y_pred"].to_numpy(dtype=float),
        calib["y_true"].to_numpy(dtype=float),
        test["y_pred"].to_numpy(dtype=float),
    )


def _observation_losses(
    task: str, y_true: np.ndarray, output: Calibrated, transform: str, level: float
) -> dict[str, np.ndarray]:
    """Per-observation losses for one calibrator, the instance-level unit."""
    n = len(y_true)
    losses = {name: np.full(n, np.nan) for name in LOSS_COLUMNS}

    if TASKS[task] == "binary":
        if output.in_set is None:
            probability = np.clip(output.y_pred, 1e-7, 1 - 1e-7)
            losses["nll"] = -(y_true * np.log(probability) + (1 - y_true) * np.log(1 - probability))
            losses["brier"] = (output.y_pred - y_true) ** 2
        else:
            losses["covered"] = output.in_set.astype(float)
            losses["set_size"] = output.set_size.astype(float)
        return losses

    losses["abs_error"] = np.abs(output.y_pred - y_true)
    losses["sq_error"] = (output.y_pred - y_true) ** 2
    if output.lower is not None:
        losses["covered"] = ((y_true >= output.lower) & (y_true <= output.upper)).astype(float)
        losses["width"] = output.upper - output.lower
        losses["interval_score"] = interval_score_per_observation(
            y_true, output.lower, output.upper, level
        )
    if output.pred_std is not None:
        mean_model = _to_model_space(output.y_pred, transform, "test prediction")
        if transform == "log":
            residual = np.log(y_true) - mean_model
            losses["nll"] = (
                0.5 * np.log(2 * np.pi * output.pred_std**2)
                + residual**2 / (2 * output.pred_std**2)
                + np.log(y_true)
            )
        else:
            residual = y_true - mean_model
            losses["nll"] = 0.5 * np.log(2 * np.pi * output.pred_std**2) + residual**2 / (
                2 * output.pred_std**2
            )
    return losses


def _calibrator_metrics(
    task: str,
    y_true: np.ndarray,
    compound_id: np.ndarray,
    output: Calibrated,
    transform: str,
    level: float,
) -> dict[str, float]:
    """Dataset-level metrics for one calibrator on one cell.

    Coverage is reported twice: ``picp`` as the raw rate, and ``coverage_error``
    as ``|picp - level|``, which is the rankable one.
    """
    metrics: dict[str, float] = {}
    if TASKS[task] == "binary" and output.in_set is not None:
        metrics["picp"] = float(np.mean(output.in_set))
        metrics["set_size"] = float(np.mean(output.set_size))
        metrics["coverage_error"] = abs(metrics["picp"] - level)
        metrics["n_observations"] = float(len(y_true))
        return metrics

    # A weakly monotone calibrator ties scores together, so accuracy is
    # recomputed per calibrator rather than shared across them.
    metrics |= accuracy_metrics(task, y_true, output.y_pred, compound_id)
    if TASKS[task] == "binary":
        metrics |= binary_uq_metrics(y_true, output.y_pred)
    else:
        if output.lower is not None:
            metrics["picp"] = float(np.mean((y_true >= output.lower) & (y_true <= output.upper)))
            metrics["mpiw"] = float(np.mean(output.upper - output.lower))
            metrics["coverage_error"] = abs(metrics["picp"] - level)
            metrics["interval_score"] = interval_score(y_true, output.lower, output.upper, level)
        if output.pred_std is not None:
            mean_model = _to_model_space(output.y_pred, transform, "test prediction")
            metrics["nll"] = (
                lognormal_nll(y_true, mean_model, output.pred_std)
                if transform == "log"
                else gaussian_nll(y_true, mean_model, output.pred_std)
            )
    metrics["n_observations"] = float(len(y_true))
    return metrics


def _distributional_frames(
    meta: dict, selection: dict, eval_set: str, final: tuple[pd.DataFrame, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """The point and scale predictions of one evaluation set."""
    return {
        name: deployed_predictions(meta, selection, eval_set, output, final=final)
        for name, output in (
            ("point", meta["point_output"]),
            ("scale", meta["scale_outputs"][0]),
        )
    }


def build_tables(
    results_root: str | Path,
    level: float = DEFAULT_LEVEL,
    classification_calibrators: tuple[str, ...] = CLASSIFICATION_CALIBRATORS,
    regression_calibrators: tuple[str, ...] = REGRESSION_CALIBRATORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate every completed cell, over every selection rule and calibrator it supports.

    Args:
        results_root: directory holding ``<task>/<head>/outer<k>/`` cells.
        level: nominal central coverage for interval and label-set calibrators.
        classification_calibrators: calibrators to run on binary tasks.
        regression_calibrators: calibrators to run on point-predicting
            regression arms.

    Returns:
        ``(observations, folds)``. ``observations`` is long over
        ``(cell, selection, calibrator, evaluation row)``; ``folds`` is long over
        ``(cell, selection, calibrator, metric)``.
    """
    observation_frames: list[pd.DataFrame] = []
    fold_records: list[dict] = []

    for metadata_path in iter_cells(results_root):
        meta = read_cell(metadata_path)
        task, transform = meta["task"], meta["target_transform"]
        is_binary = TASKS[task] == "binary"
        # A binary task's predictive is one number whatever produced it, so an
        # arm's scale is stored but scored only where an interval can carry it.
        is_distributional = bool(meta["scale_outputs"]) and not is_binary
        final = read_final(meta)

        for selection in cell_selections(meta):
            if is_distributional:
                frames = {
                    s: _distributional_frames(meta, selection, s, final) for s in ("calib", "test")
                }
                calib, test = frames["calib"]["point"], frames["test"]["point"]
                if meta["uq"] in {"mcd", "swag_sgd", "ensemble", "ens_mcd", "multiswag_sgd"}:
                    calibrators = ("none", "noise_floor")
                elif meta["uq"] in {"laplace", "vbll_posthoc", "mola", "ens_vbll_posthoc"}:
                    calibrators = ("none", "scaled")
                else:
                    raise ValueError(f"no paper calibrator declared for {meta['uq']!r}")
            else:
                calib = deployed_predictions(meta, selection, "calib", final=final)
                test = deployed_predictions(meta, selection, "test", final=final)
                calibrators = classification_calibrators if is_binary else regression_calibrators
            y_true = test["y_true"].to_numpy(dtype=float)
            compound_id = test["compound_id"].to_numpy()

            for calibrator in calibrators:
                if is_distributional:
                    output = _distributional_calibrator(
                        calibrator, frames["calib"], frames["test"], transform, level
                    )
                elif is_binary:
                    output = _classification_calibrator(calibrator, calib, test, level)
                else:
                    output = _regression_calibrator(calibrator, calib, test, transform, level)

                identity = {
                    "task": task,
                    "task_type": TASKS[task],
                    "head": meta["head"],
                    "uq": meta["uq"],
                    "selection": selection["selection"],
                    "calibration": calibrator,
                    "outer": meta["outer"],
                }
                frame = test[["row_index", "compound_id", "y_true"]].assign(
                    **identity, y_pred=output.y_pred
                )
                for name, values in _observation_losses(
                    task, y_true, output, transform, level
                ).items():
                    frame[name] = values
                observation_frames.append(frame)

                details = {
                    "primary_metric": meta["primary_metric"],
                    "direction": meta["direction"],
                    "target_transform": transform,
                    "selected_config_id": selection["selected_config_id"],
                    "deployed_iteration": selection["deployed_iteration"],
                    "n_calibration": float(len(calib)),
                    "calibrator_scale": output.scale,
                    "calibrator_offset": output.offset,
                    "calibrator_fallback": output.fallback,
                    "level": level,
                }
                calibrator_metrics = _calibrator_metrics(
                    task, y_true, compound_id, output, transform, level
                )
                for metric, value in calibrator_metrics.items():
                    fold_records.append(identity | {"metric": metric, "value": value} | details)

    if not fold_records:
        raise ValueError(f"no completed cells under {results_root}")

    return pd.concat(observation_frames, ignore_index=True), pd.DataFrame(fold_records)


# --------------------------------------------------------------------------- #
# The null reference: a predictor that ignores the structure.
# --------------------------------------------------------------------------- #

# Method key of the null reference. It has no cell, so ``arm_directory`` does not
# name it and ``read_cell`` never sees it. The head is not spelled ``null``,
# which ``pandas.read_csv`` parses back as a missing value, taking the method's
# identity with it wherever a table is read from CSV.
NULL_HEAD = "null_model"
NULL_UQ = "marginal"

# Labels reach a cell through the float32 feature cache, so a cell's stored
# ``y_true`` and the curated parquet agree to float32's relative precision. The
# tolerance is relative for that reason: the tasks holding a quantity carry
# labels two orders of magnitude above the ones holding a rate, and an absolute
# bound tight enough for the second rejects the first.
LABEL_RELATIVE_TOLERANCE = 1e-6


def _eval_frame(cell_dir: str | Path, eval_set: str) -> pd.DataFrame:
    """One evaluation set's rows at the final stage, in stored order."""
    rows = pd.read_parquet(Path(cell_dir) / EVAL_ROWS_FILE)
    rows = rows[(rows["stage"] == "final") & (rows["eval_set"] == eval_set)]
    return rows[["row_index", "compound_id", "y_true"]].reset_index(drop=True)


def _check_row_space(
    task: str,
    outer: int,
    cell_dir: str,
    rows: dict[str, pd.DataFrame],
    fold: OuterFold,
    labels: np.ndarray,
) -> None:
    """Raise unless a cell indexes the rows the curated task and the split imply.

    A cell's ``row_index`` points into the featurized arrays, which drop the rows
    that fail featurization, where the null reads the curated parquet. The two
    row spaces coincide only while nothing is dropped, and the null's fitting
    rows come from the curated one, so the agreement is checked rather than
    assumed.
    """
    for eval_set, expected in (("calib", fold.calibration_idx), ("test", fold.test_idx)):
        row_index = rows[eval_set]["row_index"].to_numpy()
        if not np.array_equal(row_index, expected):
            raise ValueError(
                f"[{task} outer{outer}] {cell_dir} indexes {len(row_index)} {eval_set} rows "
                f"where the split implies {len(expected)}, or in another order. The curated "
                f"parquet and the feature cache disagree on the row space; refeaturize the task."
            )
        stored = rows[eval_set]["y_true"].to_numpy(dtype=float)
        if not np.allclose(stored, labels[row_index], rtol=LABEL_RELATIVE_TOLERANCE, atol=0.0):
            raise ValueError(
                f"[{task} outer{outer}] {cell_dir} stores {eval_set} labels that differ from the "
                f"curated parquet by more than a relative {LABEL_RELATIVE_TOLERANCE:g}. The cell "
                f"was run against different data; regenerate it rather than comparing against it."
            )


def _null_predictive(
    task: str, transform: str, fit_labels: np.ndarray
) -> tuple[float, float | None]:
    """The constant a null model deploys, from the labels the arms fitted on.

    The point prediction comes back in original units and the scale in model
    space, which is where the panel forms its interval. A binary task has no
    scale, its predictive being the one number the positive rate gives.
    """
    if TASKS[task] == "binary":
        return float(np.mean(fit_labels)), None
    model_space = _to_model_space(fit_labels, transform, "fitting label")
    mean = float(np.mean(model_space))
    return (float(np.exp(mean)) if transform == "log" else mean), float(np.std(model_space))


def null_tables(
    results_root: str | Path,
    data_root: str | Path,
    splits_dir: str | Path,
    level: float = DEFAULT_LEVEL,
    classification_calibrators: tuple[str, ...] = CLASSIFICATION_CALIBRATORS,
    distributional_calibrators: tuple[str, ...] = DISTRIBUTIONAL_CALIBRATORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a predictor that ignores the structure, on the rows the cells were scored on.

    For each ``(task, outer fold)`` holding a cell, the mean and standard
    deviation of that fold's ``fit_idx`` labels become a constant predictive for
    every test row, and the positive rate does on a binary task. Fitting on the
    rows the arms fitted on leaves the calibration slice unseen, so the split
    conformal members of the panel keep the exchangeability their coverage rests
    on.

    The evaluation rows and their labels come from one cell of the group, so the
    null is scored against the same numbers as the arms it is compared with. It
    is emitted once per selection rule the group ran, carrying no config or
    iteration of its own.

    Args:
        results_root: directory holding ``<task>/<head>/outer<k>/`` cells.
        data_root: curated task parquets, read for the fitting labels.
        splits_dir: directory of committed split CSVs.
        level: nominal central coverage for interval and label-set calibrators.
        classification_calibrators: calibrators to run on binary tasks.
        distributional_calibrators: calibrators to run on regression tasks.

    Returns:
        ``(observations, folds)``, in the columns :func:`build_tables` returns.
    """
    by_task: dict[str, dict[int, list[dict]]] = {}
    for metadata_path in iter_cells(results_root):
        meta = read_cell(metadata_path)
        by_task.setdefault(meta["task"], {}).setdefault(meta["outer"], []).append(meta)
    if not by_task:
        raise ValueError(f"no completed cells under {results_root}")

    observation_frames: list[pd.DataFrame] = []
    fold_records: list[dict] = []

    for task, outers in sorted(by_task.items()):
        is_binary = TASKS[task] == "binary"
        transform = target_transform(task)
        metric, direction = primary_metric(task)
        frame = load_task(task, data_root)
        labels = frame["y"].to_numpy(dtype=float)
        folds = read_splits(Path(splits_dir) / f"{task}.csv").to_folds(
            frame["compound_id"].to_numpy()
        )

        for outer, metas in sorted(outers.items()):
            fold = folds[outer]
            cell_dir = metas[0]["cell_dir"]
            rows = {s: _eval_frame(cell_dir, s) for s in ("calib", "test")}
            _check_row_space(task, outer, cell_dir, rows, fold, labels)

            point, scale = _null_predictive(task, transform, labels[fold.fit_idx])
            y_true = rows["test"]["y_true"].to_numpy(dtype=float)
            compound_id = rows["test"]["compound_id"].to_numpy()
            selections = sorted({s["selection"] for meta in metas for s in cell_selections(meta)})

            if is_binary:
                calibrators = classification_calibrators
                calib, test = (rows[s].assign(y_pred=point) for s in ("calib", "test"))
            else:
                calibrators = distributional_calibrators
                frames = {
                    s: {
                        "point": rows[s].assign(y_pred=point),
                        "scale": rows[s].assign(y_pred=scale),
                    }
                    for s in ("calib", "test")
                }

            for calibrator in calibrators:
                if is_binary:
                    output = _classification_calibrator(calibrator, calib, test, level)
                else:
                    output = _distributional_calibrator(
                        calibrator, frames["calib"], frames["test"], transform, level
                    )
                losses = _observation_losses(task, y_true, output, transform, level)
                calibrator_metrics = _calibrator_metrics(
                    task, y_true, compound_id, output, transform, level
                )
                details = {
                    "primary_metric": metric,
                    "direction": direction,
                    "target_transform": transform,
                    # The null sweeps nothing, so it deploys no config and no iteration.
                    "selected_config_id": None,
                    "deployed_iteration": float("nan"),
                    "n_calibration": float(len(rows["calib"])),
                    "calibrator_scale": output.scale,
                    "calibrator_offset": output.offset,
                    "calibrator_fallback": output.fallback,
                    "level": level,
                }

                for selection in selections:
                    identity = {
                        "task": task,
                        "task_type": TASKS[task],
                        "head": NULL_HEAD,
                        "uq": NULL_UQ,
                        "selection": selection,
                        "calibration": calibrator,
                        "outer": outer,
                    }
                    observation_frames.append(
                        rows["test"][["row_index", "compound_id", "y_true"]]
                        .assign(**identity, y_pred=output.y_pred)
                        .assign(**losses)
                    )
                    for name, value in calibrator_metrics.items():
                        fold_records.append(identity | {"metric": name, "value": value} | details)

    return pd.concat(observation_frames, ignore_index=True), pd.DataFrame(fold_records)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd across outer folds, one row per ``(task, method, metric)``.

    ``sd`` is the sample standard deviation of the fold estimates. The folds
    share a dataset, so it describes their spread rather than the uncertainty of
    the benchmark mean.
    """
    keys = ["task", "task_type", *METHOD_KEYS, "metric"]
    summary = (
        folds.groupby(keys, sort=True)["value"]
        .agg(mean="mean", sd=lambda values: values.std(ddof=1), n_folds="size")
        .reset_index()
    )
    return summary.sort_values(keys).reset_index(drop=True)


def write_tables(
    observations: pd.DataFrame, folds: pd.DataFrame, out_dir: str | Path
) -> tuple[Path, Path]:
    """Write both tables to ``out_dir``, returning the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observations_path = out_dir / OBSERVATIONS_FILE
    folds_path = out_dir / FOLDS_FILE
    observations.to_parquet(observations_path, index=False)
    folds.to_parquet(folds_path, index=False)
    return observations_path, folds_path
