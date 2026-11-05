from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from admet_uq.featurize import MORGAN_DIM
from admet_uq.heads import get_head
from admet_uq.heads import mlp as mlp_head
from admet_uq.predictive import swag_sgd

TASK = "dili"
CONFIG = {
    "config_id": "cfg",
    "n_layers": 1,
    "hidden_dim": 16,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 8,
}


class SpySink:
    """Minimal sink used to create a real selected MLP checkpoint."""

    def __init__(self, features: np.ndarray):
        """Expose the evaluation features expected by the head contract."""
        self.features = features

    def record(self, predictions: np.ndarray, output: str = "mean") -> None:
        """Accept a recorded prediction without retaining it."""
        pass


def _case(rng, n_train: int = 24, n_eval: int = 9):
    n_features = MORGAN_DIM + 4
    x_train = rng.random((n_train, n_features), dtype=np.float32)
    y_train = (rng.random(n_train) > 0.5).astype(np.float32)
    x_eval = rng.random((n_eval, n_features), dtype=np.float32)
    return x_train, y_train, x_eval


@pytest.mark.parametrize(("snapshots_per_epoch", "expected"), [(1, 4), (4, 12)])
def test_it_collects_only_the_declared_post_settling_snapshots(
    tmp_path, rng, snapshots_per_epoch, expected
):
    """Settling epochs move weights but never enter the posterior moments."""
    x_train, y_train, x_eval = _case(rng)
    source = tmp_path / "source"
    get_head("mlp").fit(
        TASK,
        x_train,
        y_train,
        CONFIG,
        [SpySink(x_eval)],
        seed=5,
        n_iterations=3,
        artifacts_dir=source,
        checkpoint_at=2,
    )
    stored_config = json.loads((source / mlp_head.HEAD_CONFIG_FILE).read_text())
    result = swag_sgd.collect(
        TASK,
        x_train,
        y_train,
        x_eval,
        mlp_head.load_params(source / mlp_head.DEPLOYED_CHECKPOINT),
        stored_config,
        seed=11,
        swa_lr_at_batch_128=1.6e-2,
        settling_epochs=3,
        collection_epochs=4,
        snapshots_per_epoch=snapshots_per_epoch,
    )

    assert result.mean.shape == (len(x_eval),)
    assert result.std.shape == (len(x_eval),)
    assert np.all(np.isfinite(result.mean))
    assert np.all(result.std > 0)
    # Three batches per epoch cap a request for four at three distinct iterates.
    assert result.diagnostics["n_snapshots"] == expected
    assert result.diagnostics["parameter_displacement_l2"] > 0


def test_checkpoint_scaling_must_match_the_source_fit(tmp_path, rng):
    """A checkpoint is rejected if reconstructed training rows imply another scaler."""
    x_train, y_train, x_eval = _case(rng)
    source = tmp_path / "source"
    get_head("mlp").fit(
        TASK,
        x_train,
        y_train,
        CONFIG,
        [SpySink(x_eval)],
        n_iterations=2,
        artifacts_dir=source,
        checkpoint_at=1,
    )
    stored_config = json.loads((source / mlp_head.HEAD_CONFIG_FILE).read_text())
    stored_config["target_mean"] = 1.0

    with pytest.raises(ValueError, match="target_mean"):
        swag_sgd.collect(
            TASK,
            x_train,
            y_train,
            x_eval,
            mlp_head.load_params(source / mlp_head.DEPLOYED_CHECKPOINT),
            stored_config,
            seed=0,
            swa_lr_at_batch_128=1.6e-2,
            settling_epochs=0,
            collection_epochs=2,
        )


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "swag_from_checkpoint.py"
    spec = importlib.util.spec_from_file_location("swag_from_checkpoint_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_loads_selected_checkpoint(tmp_path, monkeypatch):
    """The runner loads the selected checkpoint."""
    runner = _load_runner()
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source = source_root / TASK / "mlp" / "outer0"
    source.mkdir(parents=True)
    (source / "deployed.npz").write_bytes(b"selected")
    (source / "last.npz").write_bytes(b"unselected-cap")
    config = CONFIG | {
        "target_mean": 0.0,
        "target_std": 1.0,
        "objective": "logistic",
    }
    (source / "head_config.json").write_text(json.dumps(config))
    origin = {
        "selected_config_id": "cfg",
        "selected_config": {key: value for key, value in CONFIG.items() if key != "config_id"},
        "selected_iteration": 7,
        "deployed_iteration": 7,
        "fold_seeds": [1, 2, 3, 4, 5, 6],
        "seed": 0,
        "split_file": "conf/splits_partition_10x5/s1/dili.csv",
        "feature_spec": "feature",
        "hpo_config_file": "conf/hpo_configs/mlp.csv",
        "member": 0,
        "member_seed": 101,
        "n_members": 10,
    }
    (source / "cell.json").write_text(json.dumps(origin))

    loaded: list[Path] = []
    monkeypatch.setattr(runner, "load_params", lambda path: loaded.append(Path(path)) or {"p": 1})
    monkeypatch.setattr(
        runner.swag_sgd,
        "collect",
        lambda *args, **kwargs: runner.swag_sgd.CollectionResult(
            mean=np.full(4, 0.5, dtype=np.float32),
            std=np.full(4, 0.1, dtype=np.float32),
            diagnostics={
                "initial_train_loss": 1.0,
                "final_train_loss": 0.9,
                "swag_mean_train_loss": 0.95,
                "parameter_displacement_l2": 0.2,
                "parameter_l2": 1.0,
                "mean_diagonal_std": 0.01,
                "max_diagonal_std": 0.02,
                "n_snapshots": 2,
            },
        ),
    )
    monkeypatch.setattr(runner, "accuracy_metrics", lambda *args: {"primary": 0.5})
    monkeypatch.setattr(runner, "verify_cell", lambda path: None)
    written = []
    monkeypatch.setattr(runner, "_write_cell", lambda *args: written.append(args[1]))

    data = {
        "X": np.zeros((6, MORGAN_DIM + 1), dtype=np.float32),
        "y": np.array([0, 1, 0, 1, 0, 1], dtype=np.float32),
        "compound_id": np.array([f"c{i}" for i in range(6)]),
    }
    fold = SimpleNamespace(
        fit_idx=np.array([0, 1]),
        calibration_idx=np.array([2, 3]),
        test_idx=np.array([4, 5]),
        train_idx=np.array([0, 1, 2, 3]),
    )
    assignment = SimpleNamespace(calibration_fraction=0.5, compound_digest="digest")
    runner._run_fold(
        TASK,
        0,
        data,
        [fold],
        assignment,
        source_root,
        target_root,
        0,
        1e-3,
        1,
        2,
        1,
        False,
    )

    assert loaded == [source / "deployed.npz"]
    assert written[0]["head_fit"]["checkpoint_source"] == str(source / "deployed.npz")
    assert written[0]["source_root"] == str(source_root)
    assert written[0]["member"] == 0
    assert written[0]["member_seed"] == 101
    assert written[0]["n_members"] == 10
