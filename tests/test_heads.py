from __future__ import annotations

from pathlib import Path

import pytest

from admet_uq.heads import HEADS, FitResult, arm_directory, get_head
from admet_uq.hpo_configs import SEARCH_SPACE, load_configs

#: Presampled configs every head searches over.
HPO_BUDGET = 32


def test_unknown_head_names_the_ones_that_exist():
    """A typo fails at the registry rather than as a missing attribute later."""
    with pytest.raises(ValueError, match="unknown head: 'mpl'"):
        get_head("mpl")


@pytest.mark.parametrize("name", HEADS)
def test_every_registered_head_meets_the_contract(name):
    """Everything the driver reads off a head module must be there."""
    head = get_head(name)
    assert isinstance(head.DEFAULT_ITERATIONS, int)
    assert callable(head.fit)
    assert callable(head.objectives)
    assert head.POINT_OUTPUT in head.OUTPUTS
    assert set(head.SCALE_OUTPUTS) <= set(head.OUTPUTS)
    assert head.POINT_OUTPUT not in head.SCALE_OUTPUTS
    assert isinstance(head.CONFIG_LIST, str)
    assert set(head.TASK_TYPES) <= {"binary", "regression"}


@pytest.mark.parametrize("name", HEADS)
def test_a_heads_method_key_names_its_own_directory(name):
    """A cell's path is derived from ``(head, uq)``, so the two must round-trip."""
    head = get_head(name)
    assert arm_directory(head.HEAD, head.UQ) == name


@pytest.mark.parametrize("name", HEADS)
def test_a_heads_config_list_carries_the_columns_it_reads(name):
    """The committed list is what a cell is run from, so a missing column fails here."""
    head = get_head(name)
    configs = load_configs(
        head.CONFIG_LIST, Path(__file__).resolve().parents[1] / "conf/hpo_configs"
    )
    assert len(configs) == HPO_BUDGET
    space = {dimension[0] for dimension in SEARCH_SPACE[head.CONFIG_LIST]}
    assert space <= set(configs[0])


def test_fit_result_defaults_to_no_metadata():
    """A head with nothing to record reports only what it ran."""
    assert FitResult(7).metadata == {}
    assert FitResult(7).n_iterations == 7
