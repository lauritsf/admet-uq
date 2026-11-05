"""Continue a selected MLP checkpoint with SGD and collect SWAG moments."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admet_uq import storage  # noqa: E402
from admet_uq.aggregate import verify_cell  # noqa: E402
from admet_uq.data import (  # noqa: E402
    TASKS,
    inverse_target,
    primary_metric,
    target_transform,
    transform_target,
)
from admet_uq.featurize import FEATURE_SPEC, featurize_task  # noqa: E402
from admet_uq.heads import SHARED_SELECTION  # noqa: E402
from admet_uq.heads.mlp import (  # noqa: E402
    DEPLOYED_CHECKPOINT,
    HEAD_CONFIG_FILE,
    load_params,
)
from admet_uq.metrics import accuracy_metrics  # noqa: E402
from admet_uq.predictive import _swag, swag_sgd  # noqa: E402
from admet_uq.splits import read_splits  # noqa: E402
from admet_uq.storage import EvalRows, PredictionBlock  # noqa: E402

log = logging.getLogger("swag_from_checkpoint")

SOURCE_HEAD = "mlp"
ARM = "mlp_swag_sgd"
OUTPUTS = ("mean", "std")
SCALE_OUTPUTS = ("std",)
_SEED_TAG = 0x53574147  # ASCII "SWAG"


def _write_cell(directory: Path, metadata: dict, eval_rows: list[EvalRows], blocks: list) -> None:
    """Write one post-training cell, with metadata last as the completion marker."""
    directory.mkdir(parents=True, exist_ok=True)
    with storage.PredictionWriter(directory / storage.PREDICTIONS_FILE) as writer:
        for block in blocks:
            writer.write(block)
    storage.write_curves(directory / storage.CURVES_FILE, [])
    storage.write_eval_rows(directory / storage.EVAL_ROWS_FILE, eval_rows)
    storage.write_metadata(directory / storage.METADATA_FILE, metadata)


def _collection_seed(origin: dict, base_seed: int) -> int:
    """Derive a method-specific continuation seed from the source refit's seed."""
    if origin.get("member_seed") is not None:
        source_seed = int(origin["member_seed"])
    elif origin.get("fold_seeds"):
        source_seed = int(origin["fold_seeds"][-1])
    else:
        source_seed = int(origin["seed"])
    return int(np.random.SeedSequence([source_seed, base_seed, _SEED_TAG]).generate_state(1)[0])


def _run_fold(
    task: str,
    outer: int,
    data: dict,
    folds,
    assignment,
    source_root: Path,
    target_root: Path,
    base_seed: int,
    swa_lr_at_batch_128: float,
    settling_epochs: int,
    collection_epochs: int,
    snapshots_per_epoch: int,
    overwrite: bool,
) -> None:
    """Continue one selected refit and write its calibration/test predictive."""
    source = storage.cell_dir(source_root, task, SOURCE_HEAD, outer)
    target = storage.cell_dir(target_root, task, ARM, outer)
    if (target / storage.METADATA_FILE).is_file() and not overwrite:
        log.info("[%s/%s/outer%d] already complete, skipping", task, ARM, outer)
        return

    checkpoint = source / DEPLOYED_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{source} carries no selected checkpoint {DEPLOYED_CHECKPOINT}")
    config_path = source / HEAD_CONFIG_FILE
    metadata_path = source / storage.METADATA_FILE
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{source} is missing its config or metadata")

    config = json.loads(config_path.read_text())
    origin = json.loads(metadata_path.read_text())
    if config.get("config_id") != origin["selected_config_id"]:
        raise ValueError(
            f"{source}: checkpoint config {config.get('config_id')!r} disagrees with "
            f"selected config {origin['selected_config_id']!r}"
        )
    if origin.get("deployed_iteration") != origin.get("selected_iteration"):
        raise ValueError(
            f"{source}: deployed iteration {origin.get('deployed_iteration')} is not the "
            f"selected iteration {origin.get('selected_iteration')}"
        )

    fold = folds[outer]
    y_raw = data["y"]
    y_fit_space = transform_target(task, y_raw)
    evaluation_index = np.concatenate([fold.calibration_idx, fold.test_idx])
    collection_seed = _collection_seed(origin, base_seed)
    started = time.time()
    result = swag_sgd.collect(
        task,
        data["X"][fold.fit_idx],
        y_fit_space[fold.fit_idx],
        data["X"][evaluation_index],
        load_params(checkpoint),
        config,
        seed=collection_seed,
        swa_lr_at_batch_128=swa_lr_at_batch_128,
        settling_epochs=settling_epochs,
        collection_epochs=collection_epochs,
        snapshots_per_epoch=snapshots_per_epoch,
    )

    n_calibration = len(fold.calibration_idx)
    split_predictions = {
        "calib": (result.mean[:n_calibration], result.std[:n_calibration]),
        "test": (result.mean[n_calibration:], result.std[n_calibration:]),
    }
    eval_rows: list[EvalRows] = []
    blocks: list[PredictionBlock] = []
    point_predictions: dict[str, np.ndarray] = {}
    for eval_set, index in (("calib", fold.calibration_idx), ("test", fold.test_idx)):
        location, spread = split_predictions[eval_set]
        if TASKS[task] == "regression":
            location = inverse_target(task, location)
        point_predictions[eval_set] = np.asarray(location)
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
        for output, values in (("mean", location), ("std", spread)):
            blocks.append(
                PredictionBlock(
                    stage="final",
                    config_id=origin["selected_config_id"],
                    inner_fold=-1,
                    eval_set=eval_set,
                    preds=np.asarray(values, dtype=np.float32)[None, :],
                    # Label the block with the source checkpoint iteration.
                    first_iteration=origin["selected_iteration"],
                    output=output,
                )
            )

    test_metrics = accuracy_metrics(
        task,
        y_raw[fold.test_idx],
        point_predictions["test"],
        data["compound_id"][fold.test_idx],
    )
    selection = {
        "deployed_iteration": origin["selected_iteration"],
        "refit_dir": str(source),
        "refit_head": SOURCE_HEAD,
        "selected_config": origin["selected_config"],
        "selected_config_id": origin["selected_config_id"],
        "selected_iteration": origin["selected_iteration"],
        "selection": SHARED_SELECTION,
        "selection_direction": primary_metric(task)[1],
        "selection_metric": primary_metric(task)[0],
        "test_metrics": test_metrics,
    }
    effective_swa_lr = (
        swa_lr_at_batch_128 * int(config["batch_size"]) / swag_sgd.REFERENCE_BATCH_SIZE
    )
    collection_metadata = {
        "checkpoint_swag": True,
        "checkpoint_source": str(checkpoint),
        "checkpoint_source_iteration": origin["selected_iteration"],
        "checkpoint_source_optimizer": "adamw",
        "collection_optimizer": "sgd",
        "collection_momentum": swag_sgd.MOMENTUM,
        "collection_learning_rate_at_batch_128": swa_lr_at_batch_128,
        "collection_learning_rate": effective_swa_lr,
        "collection_learning_rate_scaling": "linear_batch_size",
        "collection_reference_batch_size": swag_sgd.REFERENCE_BATCH_SIZE,
        "collection_weight_decay": float(config["weight_decay"]),
        "collection_settling_epochs": settling_epochs,
        "collection_epochs": collection_epochs,
        "collection_snapshots_per_epoch_requested": snapshots_per_epoch,
        "collection_snapshots_per_epoch_actual": (
            int(result.diagnostics["n_snapshots"]) // collection_epochs
        ),
        "collection_cadence": "evenly_spaced_within_epoch",
        "collection_seed": collection_seed,
        "collection_device": jax.default_backend(),
        "collection_xla_flags": os.environ.get("XLA_FLAGS", ""),
        "swag_rank": _swag.RANK,
        "swag_samples": _swag.N_SAMPLES,
        **result.diagnostics,
    }
    member_metadata = {
        key: origin[key] for key in ("member", "member_seed", "n_members") if key in origin
    }
    metadata = {
        "task": task,
        "head": SOURCE_HEAD,
        "uq": swag_sgd.UQ,
        "outputs": list(OUTPUTS),
        "scale_outputs": list(SCALE_OUTPUTS),
        "point_output": "mean",
        "outer": outer,
        "primary_metric": primary_metric(task)[0],
        "direction": primary_metric(task)[1],
        "target_transform": target_transform(task),
        "selections": {selection["selection"]: selection},
        **selection,
        "head_fit": config | collection_metadata,
        "schema_version": storage.CELL_SCHEMA_VERSION,
        "shared_sweep": None,
        "n_configs": 0,
        "n_iterations": origin["selected_iteration"],
        "seed": base_seed,
        "fold_seeds": origin.get("fold_seeds"),
        "split_file": origin.get("split_file"),
        "feature_spec": origin.get("feature_spec", FEATURE_SPEC),
        "hpo_config_file": origin.get("hpo_config_file"),
        "source_root": str(source_root),
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
        "runtime_seconds": time.time() - started,
        **member_metadata,
    }
    _write_cell(target, metadata, eval_rows, blocks)
    verify_cell(target)
    log.info(
        "[%s/%s/outer%d] %s = %.4f; loss %.5g -> %.5g; %d snapshots",
        task,
        ARM,
        outer,
        primary_metric(task)[0],
        test_metrics["primary"],
        result.diagnostics["initial_train_loss"],
        result.diagnostics["final_train_loss"],
        result.diagnostics["n_snapshots"],
    )


def main(
    task: str,
    splits_dir: Path,
    outer: int | None = None,
    data_root: Path = Path("data/tdc_admet"),
    cache_root: Path = Path("data/features"),
    source_root: Path = Path("results"),
    target_root: Path = Path("swag_sgd"),
    seed: int = 0,
    swa_lr_at_batch_128: float = swag_sgd.DEFAULT_SWA_LR_AT_BATCH_128,
    settling_epochs: int = swag_sgd.DEFAULT_SETTLING_EPOCHS,
    collection_epochs: int = swag_sgd.DEFAULT_COLLECTION_EPOCHS,
    snapshots_per_epoch: int = swag_sgd.DEFAULT_SNAPSHOTS_PER_EPOCH,
    overwrite: bool = False,
) -> None:
    """Write checkpoint-resumed SWAG cells for one task.

    Args:
        task: TDC ADMET task name.
        outer: outer fold index; every fold when omitted.
        data_root: curated-parquet root.
        cache_root: feature-cache root.
        splits_dir: persisted split directory used by the source run.
        source_root: completed run carrying selected deterministic checkpoints.
        target_root: separate results root for the checkpoint-SWAG cells.
        seed: base seed mixed with each source refit's seed.
        swa_lr_at_batch_128: constant SGD rate at batch 128; each selected
            config's effective rate scales linearly with its batch size.
        settling_epochs: SGD epochs discarded before collecting moments.
        collection_epochs: epochs and snapshots accumulated after settling.
        snapshots_per_epoch: requested evenly spaced, distinct post-update
            iterates per collection epoch; capped at that epoch's batch count.
        overwrite: rewrite a target cell whose ``cell.json`` already exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    data = featurize_task(task, data_root, cache_root)
    assignment = read_splits(Path(splits_dir) / f"{task}.csv")
    folds = assignment.to_folds(data["compound_id"])
    for fold_index in range(assignment.n_outer) if outer is None else [outer]:
        _run_fold(
            task,
            fold_index,
            data,
            folds,
            assignment,
            Path(source_root),
            Path(target_root),
            seed,
            swa_lr_at_batch_128,
            settling_epochs,
            collection_epochs,
            snapshots_per_epoch,
            overwrite,
        )


if __name__ == "__main__":
    tyro.cli(main)
