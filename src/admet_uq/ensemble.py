"""Combine stored members into equal-weight predictive mixtures.

Locations and variances are combined by moment matching in model space, then
written in the standard cell layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from admet_uq import storage
from admet_uq.aggregate import deployed_predictions, read_cell, read_final, verify_cell
from admet_uq.data import TASKS, inverse_target, transform_target
from admet_uq.heads import arm_directory
from admet_uq.storage import EvalRows, PredictionBlock

log = logging.getLogger(__name__)

#: The arm each member arm becomes once its members are mixed, as
#: ``(member head, member uq) -> ensemble uq``. The names follow the
#: constructions they implement: a deep ensemble of deterministic networks, an
#: ensemble of MC-dropout predictives, MultiSWAG, and MoLA for the last-layer
#: Laplace mixture.
ENSEMBLE_UQ = {
    "deterministic": "ensemble",
    "mcd": "ens_mcd",
    "swag_sgd": "multiswag_sgd",
    "laplace": "mola",
    "vbll_posthoc": "ens_vbll_posthoc",
}


def member_roots(members_root: str | Path, n_members: int) -> list[Path]:
    """The per-member results roots, in member order."""
    return [Path(members_root) / f"m{member:02d}" for member in range(n_members)]


def _member_outputs(meta: dict, selection: dict, eval_set: str) -> tuple[np.ndarray, np.ndarray]:
    """One member's location and spread over an evaluation set, in model space.

    The location is stored in the task's original units and the spread in model
    space, so the location is mapped back to meet it.

    A member carrying no spread reports zeros, which leaves the mixture variance
    as the disagreement between members alone.
    """
    final = read_final(meta)
    point = meta["point_output"]
    location = deployed_predictions(meta, selection, eval_set, point, final=final)
    values = transform_target(meta["task"], location["y_pred"].to_numpy())
    scales = [name for name in meta["outputs"] if name in set(meta.get("scale_outputs", ()))]
    if not scales:
        return values, np.zeros(len(location))
    spread = deployed_predictions(meta, selection, eval_set, scales[0], final=final)
    return values, spread["y_pred"].to_numpy()


def mix(locations: np.ndarray, spreads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight mixture of member predictives, matched on two moments.

    Args:
        locations: ``(n_members, n_rows)`` member locations.
        spreads: ``(n_members, n_rows)`` member spreads, zero where a member
            carries none.

    Returns:
        The mixture location and its standard deviation.
    """
    mean = locations.mean(axis=0)
    second = (spreads**2 + locations**2).mean(axis=0)
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def combine_cell(
    roots: list[Path], task: str, arm: str, outer: int, out_root: str | Path
) -> Path | None:
    """Mix one cell's members and write the ensemble cell, or skip an incomplete one.

    Returns the directory written, or ``None`` when a member is missing.
    """
    metas = []
    for root in roots:
        path = storage.cell_dir(root, task, arm, outer) / storage.METADATA_FILE
        if not path.is_file():
            return None
        metas.append(read_cell(path))

    head, uq = metas[0]["head"], metas[0]["uq"]
    selections = [meta["selections"][meta["selection"]] for meta in metas]
    target = Path(out_root) / task / arm_directory(head, ENSEMBLE_UQ[uq]) / f"outer{outer}"

    eval_rows, blocks = [], []
    for eval_set in ("calib", "test"):
        stacked = [
            _member_outputs(meta, sel, eval_set)
            for meta, sel in zip(metas, selections, strict=True)
        ]
        mean, std = mix(np.array([s[0] for s in stacked]), np.array([s[1] for s in stacked]))
        mean = inverse_target(task, mean)
        rows = deployed_predictions(metas[0], selections[0], eval_set, metas[0]["point_output"])
        eval_rows.append(
            EvalRows(
                stage="final",
                inner_fold=-1,
                eval_set=eval_set,
                row_index=rows["row_index"].to_numpy(dtype=np.int32),
                compound_id=rows["compound_id"].to_numpy(),
                y_true=rows["y_true"].to_numpy(dtype=float),
            )
        )
        for name, values in (("mean", mean), ("std", std)):
            blocks.append(
                PredictionBlock(
                    stage="final",
                    config_id=selections[0]["selected_config_id"],
                    inner_fold=-1,
                    eval_set=eval_set,
                    preds=np.asarray(values, dtype=np.float32)[None, :],
                    first_iteration=selections[0]["deployed_iteration"],
                    output=name,
                )
            )

    selection = dict(selections[0]) | {"test_metrics": None}
    metadata = dict(metas[0]) | {
        "uq": ENSEMBLE_UQ[uq],
        "outputs": ["mean", "std"],
        "scale_outputs": ["std"] if TASKS[task] == "regression" else [],
        "point_output": "mean",
        "selections": {selection["selection"]: selection},
        **selection,
        "n_members": len(roots),
        "member_uq": uq,
        # The moments are matched in the training space and the location is
        # inverted once, so ``std`` is a model-space scale like a single model's.
        "mixture_space": "model",
        # An arm fitted over a member's frozen features rather than trained with
        # it records the seed of its own fit, not the member's, so the seed is
        # taken where it is written and left out where it is not.
        "member_seeds": [meta.get("member_seed") for meta in metas],
    }
    metadata.pop("cell_dir", None)
    metadata.pop("member", None)
    metadata.pop("member_seed", None)

    target.mkdir(parents=True, exist_ok=True)
    with storage.PredictionWriter(target / storage.PREDICTIONS_FILE) as writer:
        for block in blocks:
            writer.write(block)
    storage.write_curves(target / storage.CURVES_FILE, [])
    storage.write_eval_rows(target / storage.EVAL_ROWS_FILE, eval_rows)
    storage.write_metadata(target / storage.METADATA_FILE, metadata)
    verify_cell(target)
    return target


def combine_all(
    members_root: str | Path, out_root: str | Path, n_members: int, arms: list[str]
) -> int:
    """Mix every complete cell of every arm, and report how many were written."""
    roots = member_roots(members_root, n_members)
    written = 0
    for task in TASKS:
        for arm in arms:
            for outer in range(5):
                if combine_cell(roots, task, arm, outer, out_root) is not None:
                    written += 1
        log.info("[%s] combined, %d cells so far", task, written)
    return written


def member_disagreement(roots: list[Path], task: str, arm: str, outer: int) -> dict | None:
    """How far one cell's members differ from each other on the test fold.

    The mean pairwise root-mean-square difference between member locations,
    beside the spread each member states on its own. An ensemble draws on the
    first; a member's own predictive reports the second.
    """
    metas = []
    for root in roots:
        path = storage.cell_dir(root, task, arm, outer) / storage.METADATA_FILE
        if not path.is_file():
            return None
        metas.append(read_cell(path))
    selections = [meta["selections"][meta["selection"]] for meta in metas]
    stacked = [
        _member_outputs(meta, sel, "test") for meta, sel in zip(metas, selections, strict=True)
    ]
    locations = np.array([s[0] for s in stacked])
    spreads = np.array([s[1] for s in stacked])
    pairwise = [
        float(np.sqrt(np.mean((locations[i] - locations[j]) ** 2)))
        for i in range(len(locations))
        for j in range(i + 1, len(locations))
    ]
    return {
        "task": task,
        "arm": arm,
        "outer": outer,
        "disagreement": float(np.mean(pairwise)),
        "member_spread": float(np.mean(spreads)) if spreads.any() else float("nan"),
        "prediction_scale": float(np.std(locations.mean(axis=0))),
    }


def _cli(
    members_root: Path = Path("ensemble"),
    out_root: Path = Path("ensemble_combined"),
    n_members: int = 10,
    arms: tuple[str, ...] = (
        "mlp",
        "mlp_mcd",
        "mlp_swag_sgd",
        "mlp_laplace",
        "mlp_vbll_posthoc",
    ),
) -> None:
    """Write the ensemble cells implied by a directory of member roots."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    written = combine_all(members_root, out_root, n_members, list(arms))
    log.info("wrote %d ensemble cells to %s", written, out_root)


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli)
