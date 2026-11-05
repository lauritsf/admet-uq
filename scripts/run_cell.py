"""Run nested cross-validation for one task and head."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admet_uq import storage  # noqa: E402
from admet_uq.featurize import FEATURE_SPEC, featurize_task  # noqa: E402
from admet_uq.heads import get_head, named_heads  # noqa: E402
from admet_uq.hpo_configs import load_configs  # noqa: E402
from admet_uq.nested_cv import run_cells  # noqa: E402
from admet_uq.splits import read_splits  # noqa: E402

log = logging.getLogger("run_cell")


def main(
    task: str,
    splits_dir: Path,
    head: str = "lgbm",
    outer: int | None = None,
    data_root: Path = Path("data/tdc_admet"),
    cache_root: Path = Path("data/features"),
    hpo_dir: Path = Path("conf/hpo_configs"),
    results_root: Path = Path("results"),
    n_iterations: int | None = None,
    n_jobs: int = 0,
    seed: int = 0,
    max_configs: int | None = None,
    store_inner_predictions: bool = True,
    overwrite: bool = False,
) -> None:
    """Run one outer fold, or every outer fold when ``outer`` is omitted.

    Args:
        task: TDC ADMET task name.
        head: head name, or ``shared`` for every head that shares one sweep; the
            config CSV it names must exist in ``hpo_dir``.
        outer: outer fold index.
        data_root: curated-parquet root.
        cache_root: feature-cache root (the ``FEATURE_SPEC`` subdirectory is added).
        splits_dir: directory of committed split CSVs.
        hpo_dir: directory of presampled config CSVs.
        results_root: where per-cell artifacts are written.
        n_iterations: training iterations per fit; ``None`` takes the head's default.
        n_jobs: worker threads (0 = all available).
        seed: model seed.
        max_configs: truncate the config list.
        store_inner_predictions: write the per-iteration inner-validation predictions.
        overwrite: rerun cells whose ``cell.json`` already exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    heads = named_heads(head)
    self_selecting = heads[:1]

    features = featurize_task(task, data_root, cache_root)
    assignment = read_splits(Path(splits_dir) / f"{task}.csv")
    folds = assignment.to_folds(features["compound_id"])
    config_list = get_head(heads[0]).CONFIG_LIST
    configs = load_configs(config_list, hpo_dir)
    if max_configs is not None:
        configs = configs[:max_configs]
        log.warning("smoke mode: running only %d of the presampled configs", len(configs))

    for fold_index in range(assignment.n_outer) if outer is None else [outer]:
        directories = {
            name: storage.cell_dir(results_root, task, name, fold_index) for name in heads
        }
        complete = [
            name
            for name, directory in directories.items()
            if (directory / storage.METADATA_FILE).is_file()
        ]
        if len(complete) == len(heads) and not overwrite:
            log.info("[%s/%s/outer%d] already complete, skipping", task, head, fold_index)
            continue
        if complete and not overwrite:
            # Shared sweeps rewrite all member cells together.
            log.warning(
                "[%s/%s/outer%d] rewriting %s, which the shared sweep produces anyway",
                task,
                head,
                fold_index,
                "+".join(complete),
            )

        outputs = run_cells(
            task,
            heads,
            fold_index,
            features,
            assignment,
            folds,
            configs,
            directories,
            self_selecting=self_selecting,
            n_iterations=n_iterations,
            n_jobs=n_jobs,
            seed=seed,
            store_inner_predictions=store_inner_predictions,
        )
        for name, directory in directories.items():
            outputs[name].metadata.update(
                {
                    "schema_version": storage.CELL_SCHEMA_VERSION,
                    "feature_spec": FEATURE_SPEC,
                    "split_file": str(Path(splits_dir) / f"{task}.csv"),
                    "hpo_config_file": str(Path(hpo_dir) / f"{config_list}.csv"),
                }
            )
            storage.write_curves(directory / storage.CURVES_FILE, outputs[name].curve_records)
            storage.write_eval_rows(directory / storage.EVAL_ROWS_FILE, outputs[name].eval_rows)
            # cell.json last: its presence is what marks the cell complete.
            storage.write_metadata(directory / storage.METADATA_FILE, outputs[name].metadata)
            storage.discard_partial(directory)
            log.info("[%s/%s/outer%d] wrote %s", task, name, fold_index, directory)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
