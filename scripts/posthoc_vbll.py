"""Fit post-hoc VBLL layers on selected deterministic MLP checkpoints."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admet_uq import storage
from admet_uq.aggregate import verify_cell
from admet_uq.data import TASKS, inverse_target, primary_metric, target_transform, transform_target
from admet_uq.featurize import featurize_task
from admet_uq.heads import SHARED_SELECTION
from admet_uq.heads.mlp import _transform_features, hidden, load_params
from admet_uq.metrics import accuracy_metrics
from admet_uq.predictive import vbll_posthoc
from admet_uq.splits import read_splits
from admet_uq.storage import EvalRows, PredictionBlock

log = logging.getLogger("posthoc_vbll")

SOURCE_HEAD = "mlp"
ARM = "mlp_vbll_posthoc"
UQ = "vbll_posthoc"
OUTPUTS = ("mean", "std")
SCALE_OUTPUTS = ("std",)


def _frozen(params: dict, x: np.ndarray, config: dict) -> np.ndarray:
    """Activations of the last hidden block, with dropout off."""
    return np.asarray(
        hidden(
            params,
            _transform_features(x),
            int(config["n_layers"]),
            np.float32(config["dropout"]),
        )
    )


def _write_cell(directory: Path, metadata: dict, eval_rows: list[EvalRows], blocks: list) -> None:
    """Write predictions, evaluation rows, empty curves, and metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    with storage.PredictionWriter(directory / storage.PREDICTIONS_FILE) as writer:
        for block in blocks:
            writer.write(block)
    storage.write_curves(directory / storage.CURVES_FILE, [])
    storage.write_eval_rows(directory / storage.EVAL_ROWS_FILE, eval_rows)
    storage.write_metadata(directory / storage.METADATA_FILE, metadata)


def _run_fold(
    task: str,
    outer: int,
    data: dict,
    folds,
    assignment,
    results_root: Path,
    seed: int,
    overwrite: bool,
) -> None:
    source = storage.cell_dir(results_root, task, SOURCE_HEAD, outer)
    target = storage.cell_dir(results_root, task, ARM, outer)
    if (target / storage.METADATA_FILE).is_file() and not overwrite:
        log.info("[%s/%s/outer%d] already complete, skipping", task, ARM, outer)
        return
    if not (source / "deployed.npz").is_file():
        raise FileNotFoundError(f"{source} carries no deployed checkpoint to read")

    config = json.loads((source / "head_config.json").read_text())
    origin = json.loads((source / storage.METADATA_FILE).read_text())
    params = load_params(source / "deployed.npz")
    fold = folds[outer]

    task_type = TASKS[task]
    y_raw = data["y"]
    y_fit_space = transform_target(task, y_raw)
    standardized = (y_fit_space - config["target_mean"]) / config["target_std"]

    train_features = _frozen(params, data["X"][fold.fit_idx], config)
    best = vbll_posthoc.fit_layer(
        train_features, standardized[fold.fit_idx].astype(np.float32), task_type, seed=seed
    )
    log.info(
        "[%s/%s/outer%d] layer fitted: prior %.4g wishart %.4g (negative ELBO %.5f)",
        task,
        ARM,
        outer,
        best["vbll_prior_scale"],
        best["vbll_wishart_scale"],
        best["negative_elbo"],
    )

    eval_rows, blocks = [], []
    predictions: dict[str, np.ndarray] = {}
    for eval_set, index in (("calib", fold.calibration_idx), ("test", fold.test_idx)):
        out = vbll_posthoc.predict(
            best["params"],
            _frozen(params, data["X"][index], config),
            task_type,
            target_mean=config["target_mean"],
            target_std=config["target_std"],
            seed=seed,
        )
        # Inverse-transform locations only; spreads stay in model space.
        location = inverse_target(task, out["mean"]) if task_type == "regression" else out["mean"]
        predictions[eval_set] = location
        eval_rows.append(
            EvalRows(
                stage="final",
                inner_fold=-1,
                eval_set=eval_set,
                row_index=np.asarray(index, dtype=np.int32),
                compound_id=data["compound_id"][index],
                y_true=np.asarray(y_raw[index], dtype=np.float64),
            )
        )
        for name, values in (("mean", location), ("std", out["std"])):
            blocks.append(
                PredictionBlock(
                    stage="final",
                    config_id=origin["selected_config_id"],
                    inner_fold=-1,
                    eval_set=eval_set,
                    preds=np.asarray(values, dtype=np.float32)[None, :],
                    # Label the block with the selected MLP iteration.
                    first_iteration=origin["selected_iteration"],
                    output=name,
                )
            )

    test_metrics = accuracy_metrics(
        task, y_raw[fold.test_idx], predictions["test"], data["compound_id"][fold.test_idx]
    )
    selection = {
        "deployed_iteration": origin["selected_iteration"],
        "refit_dir": f"../../{SOURCE_HEAD}/outer{outer}",
        "refit_head": SOURCE_HEAD,
        "selected_config": origin["selected_config"],
        "selected_config_id": origin["selected_config_id"],
        "selected_iteration": origin["selected_iteration"],
        "selection": SHARED_SELECTION,
        "selection_direction": primary_metric(task)[1],
        "selection_metric": primary_metric(task)[0],
        "test_metrics": test_metrics,
    }
    metadata = {
        "task": task,
        "head": SOURCE_HEAD,
        "uq": UQ,
        "outputs": list(OUTPUTS),
        "scale_outputs": list(SCALE_OUTPUTS),
        "point_output": "mean",
        "outer": outer,
        "primary_metric": primary_metric(task)[0],
        "direction": primary_metric(task)[1],
        "target_transform": target_transform(task),
        "selections": {selection["selection"]: selection},
        **selection,
        "head_fit": config
        | {
            "vbll_prior_scale": best["vbll_prior_scale"],
            "vbll_wishart_scale": best["vbll_wishart_scale"],
            "vbll_negative_elbo": best["negative_elbo"],
            "vbll_prior_grid": list(vbll_posthoc.PRIOR_GRID),
            "vbll_wishart_grid": list(vbll_posthoc.WISHART_GRID),
            "vbll_posthoc_steps": vbll_posthoc.STEPS,
            "vbll_posthoc_learning_rate": vbll_posthoc.LEARNING_RATE,
        },
        "schema_version": storage.CELL_SCHEMA_VERSION,
        "shared_sweep": None,
        "n_configs": 0,
        "n_iterations": origin["selected_iteration"],
        "seed": seed,
        "fold_seeds": origin.get("fold_seeds"),
        "split_file": origin.get("split_file"),
        "feature_spec": origin.get("feature_spec"),
        "hpo_config_file": origin.get("hpo_config_file"),
        "calibration_fraction_target": assignment.calibration_fraction,
        "calibration_fraction_realised": len(fold.calibration_idx) / len(fold.train_idx),
        "n_rows": {
            "outer_train": len(fold.train_idx),
            "fit": len(fold.fit_idx),
            "calibration": len(fold.calibration_idx),
            "test": len(fold.test_idx),
        },
        "inner_predictions_stored": False,
        "compound_digest": assignment.compound_digest,
    }
    _write_cell(target, metadata, eval_rows, blocks)
    verify_cell(target)
    log.info(
        "[%s/%s/outer%d] %s = %.4f",
        task,
        ARM,
        outer,
        primary_metric(task)[0],
        test_metrics["primary"],
    )


def main(
    task: str,
    splits_dir: Path,
    outer: int | None = None,
    data_root: Path = Path("data/tdc_admet"),
    cache_root: Path = Path("data/features"),
    results_root: Path = Path("results"),
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    """Write the post-hoc VBLL cells of one task.

    Args:
        task: TDC ADMET task name.
        outer: outer fold index; every fold when omitted.
        data_root: curated-parquet root.
        cache_root: feature-cache root.
        splits_dir: directory of committed split CSVs.
        results_root: where the cells are read from and written to.
        seed: seeds the layer's initialisation and the binary predictive's draws.
        overwrite: rewrite cells whose ``cell.json`` already exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    data = featurize_task(task, data_root, cache_root)
    assignment = read_splits(Path(splits_dir) / f"{task}.csv")
    folds = assignment.to_folds(data["compound_id"])
    for fold_index in range(assignment.n_outer) if outer is None else [outer]:
        _run_fold(task, fold_index, data, folds, assignment, Path(results_root), seed, overwrite)


if __name__ == "__main__":
    tyro.cli(main)
