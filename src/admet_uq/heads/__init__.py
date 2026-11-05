"""Head definitions and lazy registry.

A :class:`Head` specifies its outputs, selection objectives, configuration
list, supported task types, and fitting function. Modules are imported on
demand.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module

# The module each registered head is declared in, under the name ``SPEC``. A
# neural arm is declared beside the method it implements rather than in a module
# of its own, so one file holds the whole of it.
_DECLARED_IN = {
    "lgbm": "admet_uq.heads.lgbm",
    "mlp": "admet_uq.heads.mlp",
    "mlp_mcd": "admet_uq.predictive.mc_dropout",
    "mlp_laplace": "admet_uq.predictive.laplace",
}

HEADS = tuple(_DECLARED_IN)

#: Heads that train one network from one config list and read it differently, so
#: one sweep serves all of them and each is still written as its own cell. The
#: first is the deterministic head, whose selection the others deploy.
SHARED_SWEEP = ("mlp", "mlp_mcd", "mlp_laplace")

#: The head name that stands for all of :data:`SHARED_SWEEP` at an entrypoint.
SHARED = "shared"


def named_heads(head: str) -> list[str]:
    """The heads an entrypoint's ``--head`` runs, which is one unless it is a group."""
    return list(SHARED_SWEEP) if head == SHARED else [head]


# The uq label of an arm carrying no UQ method: the bare point predictor.
DETERMINISTIC = "deterministic"

# The selection rule of a head selected on its task's leaderboard metric, which
# is what a cell recorded before a head could name more than one rule.
DEFAULT_SELECTION = "leaderboard"

#: The selection of an arm deploying the network the group's deterministic head
#: chose, rather than one it chose for itself.
SHARED_SELECTION = "shared"


def arm_directory(head: str, uq: str) -> str:
    """The path level a cell of this method key is written to."""
    return head if uq == DETERMINISTIC else f"{head}_{uq}"


@dataclass(frozen=True)
class FitResult:
    """What a fit reports about itself once it is done.

    ``n_iterations`` is what the fit actually ran, which can fall below the
    budget it was given. ``metadata`` carries what the head needs recorded in
    ``cell.json`` to reproduce it.
    """

    n_iterations: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Head:
    """One method the benchmark runs: what it fits, records and is selected on.

    The fields carry the names the driver, ``cell.json`` and the results tables
    use, so a head reads the same in the code as in the artifacts it produces.
    """

    HEAD: str
    UQ: str
    DEFAULT_ITERATIONS: int
    OUTPUTS: tuple[str, ...]
    SCALE_OUTPUTS: tuple[str, ...]
    POINT_OUTPUT: str
    CONFIG_LIST: str
    TASK_TYPES: tuple[str, ...]
    fit: Callable[..., FitResult]
    objectives: Callable[[str, object], dict]
    #: What this head reads a trained network through, for the backbone to
    #: combine when several heads share one sweep. ``None`` for a head that
    #: shares nothing.
    ARM: object | None = None


def get_head(name: str) -> Head:
    """The registered head ``name``.

    The module it is declared in is imported here rather than at package import,
    since a head can carry a heavy dependency and one run uses one head.
    """
    if name not in _DECLARED_IN:
        raise ValueError(f"unknown head: {name!r}; expected one of {HEADS}")
    return import_module(_DECLARED_IN[name]).SPEC


def group_fit(names: Sequence[str]) -> Callable[..., FitResult]:
    """One fit that trains a network once and records every named head's outputs.

    The heads must share a backbone, which is what lets one training pass serve
    them all; the backbone builds the combined arm from theirs, and the fit
    records under ``<uq>:<output>`` for a
    :class:`~admet_uq.curves.GroupSink` to split back out.
    """
    # Import lazily so non-neural heads do not load the neural dependencies.
    from admet_uq.heads.mlp import group_fit as build

    return build(tuple((get_head(name).UQ, get_head(name).ARM) for name in names))
