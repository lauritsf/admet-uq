"""Presampled HPO config lists, read back at eval time.

Configs are drawn once per head from a fixed search space with a low-discrepancy
Sobol sequence, keyed by a stable ``config_id``, and checked into
``conf/hpo_configs/<head>.csv``. Every config is evaluated to completion.

The search space covers the head's own hyperparameters.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

# Per-head search space: (name, kind, low, high). Regenerate the checked-in
# config CSV if the space changes.
SEARCH_SPACE: dict[str, list[tuple[str, str, float, float]]] = {
    "lgbm": [
        ("learning_rate", "loguniform", 0.01, 0.3),
        ("num_leaves", "logint", 15, 255),
        ("min_child_samples", "int", 5, 100),
        ("feature_fraction", "uniform", 0.4, 1.0),
        ("bagging_fraction", "uniform", 0.5, 1.0),
        ("lambda_l1", "loguniform", 1e-3, 10.0),
        ("lambda_l2", "loguniform", 1e-3, 10.0),
    ],
    # Dropout is bounded away from zero because these configs are also used for
    # MC dropout.
    "mlp": [
        ("n_layers", "level", 1, 4),
        ("hidden_dim", "pow2", 128, 1024),
        ("dropout", "uniform", 0.10, 0.75),
        ("learning_rate", "loguniform", 1e-5, 1e-3),
        ("weight_decay", "loguniform", 1e-6, 1e-2),
        ("batch_size", "pow2", 16, 256),
    ],
}

def _map_dim(unit: float, kind: str, low: float, high: float):
    """Map a unit-cube coordinate ``unit`` in [0,1] to a search-space value."""
    if kind == "uniform":
        return float(low + unit * (high - low))
    if kind == "loguniform":
        return float(np.exp(np.log(low) + unit * (np.log(high) - np.log(low))))
    if kind == "int":
        return int(round(low + unit * (high - low)))
    if kind == "logint":
        return int(round(np.exp(np.log(low) + unit * (np.log(high) - np.log(low)))))
    if kind == "pow2":
        return _pick(unit, [int(low) << step for step in range(int(np.log2(high / low)) + 1)])
    if kind == "level":
        return _pick(unit, list(range(int(low), int(high) + 1)))
    raise ValueError(f"unknown dim kind: {kind!r}")


def _pick(unit: float, levels: list[int]) -> int:
    """Choose from ``levels`` by equal shares of [0,1].

    ``int`` rounds instead, which halves the share of the two end values.
    """
    return levels[min(int(unit * len(levels)), len(levels) - 1)]


def sample_configs(head: str, n: int = 32, seed: int = 0) -> pd.DataFrame:
    """Draw ``n`` Sobol-sampled configs for ``head`` as a DataFrame with ``config_id``."""
    space = SEARCH_SPACE[head]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol balance warning when n is not a power of 2
        points = qmc.Sobol(d=len(space), scramble=True, seed=seed).random(n)
    rows = []
    for config_i in range(n):
        config = {"config_id": f"{head}_{config_i:03d}"}
        for dim_i, (name, kind, low, high) in enumerate(space):
            config[name] = _map_dim(points[config_i, dim_i], kind, low, high)
        rows.append(config)
    return pd.DataFrame(rows)


def write_configs(head: str, out_dir: str | Path, n: int = 32, seed: int = 0) -> Path:
    """Write the config list to ``<out_dir>/<head>.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{head}.csv"
    sample_configs(head, n, seed).to_csv(path, index=False)
    return path


def load_configs(head: str, conf_dir: str | Path) -> list[dict]:
    """Read ``<conf_dir>/<head>.csv`` as one dict per config."""
    df = pd.read_csv(Path(conf_dir) / f"{head}.csv")
    return df.to_dict("records")


def _cli(head: str, n: int = 32, seed: int = 0, out_dir: Path = Path("conf/hpo_configs")) -> None:
    """Generate and write a head's presampled config list."""
    path = write_configs(head, out_dir, n, seed)
    print(f"wrote {n} {head} configs -> {path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli)
