from __future__ import annotations

import numpy as np
import pytest

from admet_uq import storage
from admet_uq.aggregate import deployed_predictions, read_cell, read_final
from admet_uq.data import target_transform
from admet_uq.ensemble import combine_cell, member_roots
from admet_uq.heads import arm_directory
from admet_uq.storage import EvalRows, PredictionBlock, PredictionWriter

IDENTITY_TASK = "caco2_wang"
LOG_TASK = "half_life_obach"
N_ROWS = 16


def write_member(root, task: str, uq: str, location, spread=None) -> None:
    """One member cell holding a single deployed iteration."""
    cell = storage.cell_dir(root, task, arm_directory("mlp", uq), 0)
    rows = [
        EvalRows(
            "final",
            -1,
            eval_set,
            np.arange(N_ROWS, dtype=np.int32),
            np.array([f"c{i:03d}" for i in range(N_ROWS)]),
            np.linspace(1.0, 2.0, N_ROWS),
        )
        for eval_set in ("calib", "test")
    ]
    outputs = ["mean"] if spread is None else ["mean", "std"]
    with PredictionWriter(cell / storage.PREDICTIONS_FILE) as writer:
        for entry in rows:
            for name, values in zip(outputs, (location, spread), strict=False):
                writer.write(
                    PredictionBlock(
                        "final",
                        "cfg",
                        -1,
                        entry.eval_set,
                        np.asarray(values, dtype=np.float32)[None, :],
                        first_iteration=0,
                        output=name,
                    )
                )

    storage.write_curves(cell / storage.CURVES_FILE, [])
    storage.write_eval_rows(cell / storage.EVAL_ROWS_FILE, rows)
    selection = {
        "selection": "shared",
        "selected_config_id": "cfg",
        "deployed_iteration": 0,
        "selected_iteration": 0,
    }
    storage.write_metadata(
        cell / storage.METADATA_FILE,
        {
            "task": task,
            "head": "mlp",
            "uq": uq,
            "outer": 0,
            "primary_metric": "mae",
            "direction": "minimize",
            "target_transform": target_transform(task),
            "outputs": outputs,
            "scale_outputs": [] if spread is None else ["std"],
            "point_output": "mean",
            "schema_version": 3,
            "member_seed": 7,
            "selections": {"shared": selection},
            **selection,
        },
    )


def read_output(cell_dir, name: str, eval_set: str = "test") -> np.ndarray:
    """One output of a written cell at its deployed iteration."""
    meta = read_cell(cell_dir / storage.METADATA_FILE)
    selection = meta["selections"][meta["selection"]]
    final = read_final(meta)
    frame = deployed_predictions(meta, selection, eval_set, name, final=final)
    return frame["y_pred"].to_numpy()


@pytest.mark.parametrize("task", [IDENTITY_TASK, LOG_TASK])
def test_a_one_member_ensemble_reproduces_its_member(tmp_path, task):
    """One member mixed with nothing is that member, location and spread alike."""
    location = np.linspace(0.5, 3.0, N_ROWS)
    spread = np.linspace(0.1, 0.4, N_ROWS)
    members = tmp_path / "members"
    write_member(members / "m00", task, "swag_sgd", location, spread)

    target = combine_cell(
        member_roots(members, 1), task, "mlp_swag_sgd", 0, tmp_path / "combined"
    )

    assert target is not None
    np.testing.assert_allclose(read_output(target, "mean"), location, rtol=1e-6)
    np.testing.assert_allclose(read_output(target, "std"), spread, rtol=1e-6)


def test_checkpoint_swag_members_become_checkpoint_multiswag(tmp_path):
    """Checkpoint SWAG members map to checkpoint MultiSWAG."""
    location = np.linspace(0.5, 3.0, N_ROWS)
    spread = np.linspace(0.1, 0.4, N_ROWS)
    members = tmp_path / "members"
    write_member(members / "m00", IDENTITY_TASK, "swag_sgd", location, spread)

    target = combine_cell(
        member_roots(members, 1),
        IDENTITY_TASK,
        "mlp_swag_sgd",
        0,
        tmp_path / "combined",
    )

    assert target == tmp_path / "combined" / IDENTITY_TASK / "mlp_multiswag_sgd" / "outer0"
    assert read_cell(target / storage.METADATA_FILE)["member_uq"] == "swag_sgd"


@pytest.mark.parametrize("task", [IDENTITY_TASK, LOG_TASK])
def test_identical_members_do_not_inflate_the_spread(tmp_path, task):
    """Members that agree contribute no disagreement term."""
    location = np.linspace(0.5, 3.0, N_ROWS)
    spread = np.full(N_ROWS, 0.25)
    members = tmp_path / "members"
    for member in range(3):
        write_member(members / f"m{member:02d}", task, "swag_sgd", location, spread)

    target = combine_cell(
        member_roots(members, 3), task, "mlp_swag_sgd", 0, tmp_path / "combined"
    )

    np.testing.assert_allclose(read_output(target, "mean"), location, rtol=1e-6)
    np.testing.assert_allclose(read_output(target, "std"), spread, rtol=1e-6)


def test_a_log_target_mixes_in_the_training_space(tmp_path):
    """Two log-target members average in log space, not in reported units."""
    first, second = 0.5, 1.5
    members = tmp_path / "members"
    write_member(members / "m00", LOG_TASK, "deterministic", np.full(N_ROWS, np.exp(first)))
    write_member(members / "m01", LOG_TASK, "deterministic", np.full(N_ROWS, np.exp(second)))

    target = combine_cell(member_roots(members, 2), LOG_TASK, "mlp", 0, tmp_path / "combined")

    expected = np.exp((first + second) / 2)
    np.testing.assert_allclose(read_output(target, "mean"), expected, rtol=1e-5)
    # The spread of a deterministic arm is the members' disagreement, which is
    # half their separation in the space they were trained in.
    np.testing.assert_allclose(read_output(target, "std"), (second - first) / 2, rtol=1e-5)


def test_a_deterministic_mixture_records_the_space_it_was_matched_in(tmp_path):
    """A combined cell says which space its scale belongs to."""
    members = tmp_path / "members"
    for member in range(2):
        write_member(members / f"m{member:02d}", LOG_TASK, "deterministic", np.full(N_ROWS, 2.0))

    target = combine_cell(member_roots(members, 2), LOG_TASK, "mlp", 0, tmp_path / "combined")

    assert read_cell(target / storage.METADATA_FILE)["mixture_space"] == "model"


def test_a_missing_member_is_skipped_rather_than_partly_mixed(tmp_path):
    """A cell one member short returns nothing at all."""
    members = tmp_path / "members"
    write_member(
        members / "m00", IDENTITY_TASK, "swag_sgd", np.ones(N_ROWS), np.ones(N_ROWS)
    )

    assert (
        combine_cell(member_roots(members, 2), IDENTITY_TASK, "mlp_swag_sgd", 0, tmp_path)
        is None
    )
