"""Train ensemble members from selections in a completed nested-CV run.

Each member refits the selected configuration and iteration under its own seed
and is written to ``<members-root>/mNN``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admet_uq import storage  # noqa: E402
from admet_uq.featurize import FEATURE_SPEC, featurize_task  # noqa: E402
from admet_uq.heads import get_head, named_heads  # noqa: E402
from admet_uq.hpo_configs import load_configs  # noqa: E402
from admet_uq.nested_cv import member_seeds, run_member_cells  # noqa: E402
from admet_uq.splits import read_splits  # noqa: E402

log = logging.getLogger("run_members")


def _source_selection(source_root: Path, task: str, head: str, outer: int) -> dict:
    """Read the selected configuration and iteration for a source cell."""
    path = storage.cell_dir(source_root, task, head, outer) / storage.METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no source cell to inherit from: {path}")
    record = json.loads(path.read_text())
    return record["selections"][record["selection"]]


def main(
    task: str,
    splits_dir: Path,
    head: str = "shared",
    outer: int | None = None,
    members: int = 10,
    data_root: Path = Path("data/tdc_admet"),
    cache_root: Path = Path("data/features"),
    hpo_dir: Path = Path("conf/hpo_configs"),
    source_root: Path = Path("results"),
    members_root: Path = Path("ensemble"),
    n_iterations: int | None = None,
    n_jobs: int = 0,
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    """Run every member of one outer fold, or of every outer fold.

    Args:
        task: TDC ADMET task name.
        head: head name, or ``shared`` for every head that shares one sweep.
        outer: outer fold index; omitted runs them all.
        members: how many members to train.
        data_root: curated-parquet root.
        cache_root: feature-cache root (the ``FEATURE_SPEC`` subdirectory is added).
        splits_dir: directory of committed split CSVs.
        hpo_dir: directory of presampled config CSVs.
        source_root: the completed run whose selections the members inherit.
        members_root: holds one results root per member, ``m00`` upwards.
        n_iterations: training iterations per fit; ``None`` takes the head's default.
        n_jobs: worker threads (0 = all available).
        seed: the ensemble's base seed, which the member seeds are spawned from.
        overwrite: rerun members whose ``cell.json`` already exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    heads = named_heads(head)
    features = featurize_task(task, data_root, cache_root)
    assignment = read_splits(Path(splits_dir) / f"{task}.csv")
    folds = assignment.to_folds(features["compound_id"])
    config_list = get_head(heads[0]).CONFIG_LIST
    by_id = {config["config_id"]: config for config in load_configs(config_list, hpo_dir)}
    seeds = member_seeds(seed, members)

    for fold_index in range(assignment.n_outer) if outer is None else [outer]:
        selections = {
            name: _source_selection(source_root, task, name, fold_index) for name in heads
        }
        config = by_id[selections[heads[0]]["selected_config_id"]]
        for member in range(members):
            root = Path(members_root) / f"m{member:02d}"
            directories = {name: storage.cell_dir(root, task, name, fold_index) for name in heads}
            if not overwrite and all(
                (directory / storage.METADATA_FILE).is_file() for directory in directories.values()
            ):
                log.info(
                    "[%s/%s/outer%d] member %d already complete, skipping",
                    task,
                    head,
                    fold_index,
                    member,
                )
                continue

            outputs = run_member_cells(
                task,
                heads,
                fold_index,
                features,
                assignment,
                folds,
                config,
                directories,
                selections,
                member=member,
                member_seed=seeds[member],
                n_iterations=n_iterations,
                n_jobs=n_jobs,
                seed=seed,
            )
            for name, directory in directories.items():
                outputs[name].metadata.update(
                    {
                        "schema_version": storage.CELL_SCHEMA_VERSION,
                        "feature_spec": FEATURE_SPEC,
                        "split_file": str(Path(splits_dir) / f"{task}.csv"),
                        "hpo_config_file": str(Path(hpo_dir) / f"{config_list}.csv"),
                        "source_root": str(source_root),
                        "n_members": members,
                    }
                )
                storage.write_curves(directory / storage.CURVES_FILE, outputs[name].curve_records)
                storage.write_eval_rows(directory / storage.EVAL_ROWS_FILE, outputs[name].eval_rows)
                # cell.json last: its presence is what marks the cell complete.
                storage.write_metadata(directory / storage.METADATA_FILE, outputs[name].metadata)
                storage.discard_partial(directory)
            log.info("[%s/%s/outer%d] wrote member %d to %s", task, head, fold_index, member, root)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
