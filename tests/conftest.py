"""Shared builders for the statistics tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator; every test in this suite must be deterministic."""
    return np.random.default_rng(20260727)


def make_folds(
    values: dict[str, dict[str, float | list[float]]],
    metric: str = "primary",
    direction: str | dict[str, str] = "maximize",
    task_type: str = "binary",
) -> pd.DataFrame:
    """Build a long ``folds``-shaped frame from ``{task: {calibrator: value(s)}}``.

    A scalar value is one outer fold; a list is one entry per outer fold.
    ``direction`` is a single string or a per-task mapping (a mixed panel).
    """
    rows = []
    for task, calibrators in values.items():
        task_direction = direction[task] if isinstance(direction, dict) else direction
        for calibrator, value in calibrators.items():
            per_fold = value if isinstance(value, (list, tuple)) else [value]
            for outer, fold_value in enumerate(per_fold):
                rows.append(
                    {
                        "task": task,
                        "task_type": task_type,
                        "head": "lgbm",
                        "uq": "deterministic",
                        "selection": "leaderboard",
                        "calibration": calibrator,
                        "outer": outer,
                        "metric": metric,
                        "value": fold_value,
                        "direction": task_direction,
                    }
                )
    return pd.DataFrame(rows)
