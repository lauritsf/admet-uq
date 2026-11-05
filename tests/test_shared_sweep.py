from __future__ import annotations

import numpy as np
import pytest

from admet_uq import storage
from admet_uq.aggregate import cell_selections, deployed_predictions, read_cell
from admet_uq.featurize import MORGAN_DIM
from admet_uq.heads import SHARED_SELECTION, SHARED_SWEEP
from admet_uq.heads import mlp as mlp_head
from admet_uq.nested_cv import run_cells
from admet_uq.splits import OuterFold, SplitAssignment

BINARY_TASK = "dili"
REGRESSION_TASK = "caco2_wang"
N_ITERATIONS = 4

CONFIGS = [
    {
        "config_id": f"cfg{i}",
        "n_layers": 1,
        "hidden_dim": 16,
        "dropout": 0.3,
        "learning_rate": rate,
        "weight_decay": 1e-5,
        "batch_size": 8,
    }
    for i, rate in enumerate((1e-2, 3e-3))
]


def make_case(rng, task: str, n_rows: int = 120):
    """Features, labels and a fold laid out directly, as the split file would give it."""
    n_features = MORGAN_DIM + 4
    compound_id = np.array([f"c{i:04d}" for i in range(n_rows)])
    y = (
        (rng.random(n_rows) > 0.5).astype(np.float64)
        if task == BINARY_TASK
        else rng.normal(0.0, 1.0, n_rows)
    )
    features = {
        "X": rng.random((n_rows, n_features), dtype=np.float32),
        "y": y,
        "compound_id": compound_id,
    }
    index = np.arange(n_rows)
    test_idx, calibration_idx, fit_idx = index[:30], index[30:50], index[50:]
    fold = OuterFold(
        train_idx=index[30:],
        test_idx=test_idx,
        calibration_idx=calibration_idx,
        fit_idx=fit_idx,
        inner=[(fit_idx[:35], fit_idx[35:]), (fit_idx[35:], fit_idx[:35])],
    )
    assignment = SplitAssignment(
        task=task,
        compound_ids=list(compound_id),
        cluster=np.zeros(n_rows, dtype=int),
        outer=np.zeros(n_rows, dtype=int),
        inner=np.zeros((n_rows, 1), dtype=int),
        n_outer=1,
        n_inner=2,
        seed=0,
        cutoff=0.5,
        calibration_fraction=0.2,
        rdkit_version="test",
        compound_digest="digest",
    )
    return features, assignment, [fold]


def run_group(tmp_path, rng, task, self_selecting):
    """Run the group on one outer fold and write every head's artifacts."""
    features, assignment, folds = make_case(rng, task)
    directories = {name: storage.cell_dir(tmp_path, task, name, 0) for name in SHARED_SWEEP}
    outputs = run_cells(
        task,
        SHARED_SWEEP,
        0,
        features,
        assignment,
        folds,
        CONFIGS,
        directories,
        self_selecting=self_selecting,
        n_iterations=N_ITERATIONS,
    )
    for name, directory in directories.items():
        storage.write_curves(directory / storage.CURVES_FILE, outputs[name].curve_records)
        storage.write_eval_rows(directory / storage.EVAL_ROWS_FILE, outputs[name].eval_rows)
        # The schema version is what tells a reader an output column is there to
        # filter on, and the entrypoint stamps it beside the driver's metadata.
        outputs[name].metadata["schema_version"] = storage.CELL_SCHEMA_VERSION
        storage.write_metadata(directory / storage.METADATA_FILE, outputs[name].metadata)
    return outputs, directories


def test_every_head_of_the_group_is_written_as_its_own_cell(tmp_path, rng):
    """One sweep, four cells, each holding the four files a cell always holds."""
    _, directories = run_group(tmp_path, rng, BINARY_TASK, SHARED_SWEEP[:1])

    assert sorted(directories) == sorted(SHARED_SWEEP)
    for directory in directories.values():
        for name in (
            storage.CURVES_FILE,
            storage.PREDICTIONS_FILE,
            storage.EVAL_ROWS_FILE,
            storage.METADATA_FILE,
        ):
            assert (directory / name).is_file()


def test_an_arm_deploys_the_network_the_deterministic_head_chose(tmp_path, rng):
    """The headline is the standard post-hoc construction: same config, same epoch."""
    outputs, _ = run_group(tmp_path, rng, BINARY_TASK, SHARED_SWEEP[:1])

    chosen = outputs["mlp"].metadata["selections"]["leaderboard"]
    for name in SHARED_SWEEP[1:]:
        shared = outputs[name].metadata["selections"][SHARED_SELECTION]
        assert shared["selected_config_id"] == chosen["selected_config_id"]
        assert shared["selected_iteration"] == chosen["selected_iteration"]
        # The shared rule owns the cell, so it is what the flat fields describe.
        assert outputs[name].metadata["selection"] == SHARED_SELECTION


def test_non_self_selecting_arm_records_only_shared_selection(tmp_path, rng):
    """A non-self-selecting arm records only the shared selection."""
    outputs, _ = run_group(tmp_path, rng, BINARY_TASK, SHARED_SWEEP[:1])

    assert list(outputs["mlp"].metadata["selections"]) == ["leaderboard"]
    for name in SHARED_SWEEP[1:]:
        assert list(outputs[name].metadata["selections"]) == [SHARED_SELECTION]
        assert outputs[name].curve_records == []
        assert outputs[name].metadata["inner_predictions_stored"] is False


def test_on_the_subset_every_arm_also_chooses_for_itself(tmp_path, rng):
    """The secondary axis adds each arm's own rules beside the shared one."""
    outputs, _ = run_group(tmp_path, rng, BINARY_TASK, SHARED_SWEEP)

    for name in SHARED_SWEEP[1:]:
        assert list(outputs[name].metadata["selections"]) == [
            SHARED_SELECTION,
            "leaderboard",
            "nll",
        ]
        assert outputs[name].curve_records
        assert outputs[name].metadata["inner_predictions_stored"] is True


@pytest.mark.parametrize("task", [BINARY_TASK, REGRESSION_TASK])
def test_every_selection_reads_back_through_the_aggregation_path(tmp_path, rng, task):
    """Several refits land in one cell, so a rule is found by its config and iteration."""
    _, directories = run_group(tmp_path, rng, task, SHARED_SWEEP)

    for directory in directories.values():
        meta = read_cell(directory / storage.METADATA_FILE)
        selections = cell_selections(meta)
        assert selections
        for selection in selections:
            for eval_set in ("calib", "test"):
                frame = deployed_predictions(meta, selection, eval_set)
                assert len(frame) > 0
                assert np.all(np.isfinite(frame["y_pred"].to_numpy()))


def test_the_group_trains_the_network_the_deterministic_head_trains_alone(tmp_path, rng):
    """One training pass serves every arm, so no arm's weights move by being grouped.

    Weights rather than predictions: a carry of a different arity fuses
    differently, which moves the last bits of a recorded prediction while leaving
    the trajectory it was read from identical.
    """
    features, assignment, folds = make_case(rng, BINARY_TASK)
    weights = []
    for names in (SHARED_SWEEP, SHARED_SWEEP[:1]):
        root = tmp_path / f"group{len(names)}"
        directories = {name: storage.cell_dir(root, BINARY_TASK, name, 0) for name in names}
        run_cells(
            BINARY_TASK,
            names,
            0,
            features,
            assignment,
            folds,
            CONFIGS,
            directories,
            self_selecting=names[:1],
            n_iterations=N_ITERATIONS,
        )
        weights.append(mlp_head.load_params(directories["mlp"] / mlp_head.LAST_CHECKPOINT))

    assert sorted(weights[0]) == sorted(weights[1])
    for name in weights[0]:
        np.testing.assert_array_equal(np.asarray(weights[0][name]), np.asarray(weights[1][name]))
