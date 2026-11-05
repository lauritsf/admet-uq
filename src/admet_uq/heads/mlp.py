"""JAX MLP head with per-epoch prediction trajectories.

Training uses nested ``lax.scan`` loops and padded arrays to reuse compiled
programs. Morgan counts are log-transformed, and regression targets are
standardized within each fit. UQ methods implement the :class:`Arm` interface.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax, random

from admet_uq.curves import (
    ARM_SEPARATOR,
    EvalSink,
    Objective,
    leaderboard_objective,
    nll_objective,
)
from admet_uq.data import TASKS, primary_metric
from admet_uq.featurize import MORGAN_DIM
from admet_uq.heads import DEFAULT_SELECTION, DETERMINISTIC, FitResult, Head
from admet_uq.storage import EvalRows

HEAD = "mlp"
UQ = DETERMINISTIC
DEFAULT_ITERATIONS = 300
OUTPUTS = ("mean",)
SCALE_OUTPUTS = ()
POINT_OUTPUT = "mean"
CONFIG_LIST = "mlp"
TASK_TYPES = ("binary", "regression")

#: What an arm carrying a predictive distribution records: a location and the
#: spread around it. The spread stays in the modelling space, which is what
#: naming it as a scale output tells storage.
PREDICTIVE_SCALE = "std"
PREDICTIVE_OUTPUTS = (POINT_OUTPUT, PREDICTIVE_SCALE)

DEPLOYED_CHECKPOINT = "deployed.npz"
LAST_CHECKPOINT = "last.npz"
HEAD_CONFIG_FILE = "head_config.json"

#: Evaluation rows are padded up to a multiple of this, so that folds differing
#: by a few rows share a compiled program.
EVAL_BLOCK = 128


def objectives(task: str, rows: EvalRows) -> dict[str, Objective]:
    """Select on the leaderboard metric of the task."""
    return {DEFAULT_SELECTION: leaderboard_objective(task, rows, POINT_OUTPUT)}


@dataclass(frozen=True)
class Fit:
    """What an arm can read about the fit it belongs to.

    Every field is fixed for the whole fit, so an arm is handed this once and
    then only the current weights, a key, and whatever state it carries itself.
    ``hyper`` holds the arm's own hyperparameters, read from the config row.
    """

    n_layers: int
    binary: bool
    objective: str
    x_train: jnp.ndarray
    y_train: jnp.ndarray
    valid: jnp.ndarray
    dropout: jnp.ndarray
    target_mean: jnp.ndarray
    target_std: jnp.ndarray
    hyper: dict

    def location(self, raw):
        """Map a network output into the head's output space."""
        return jax.nn.sigmoid(raw) if self.binary else raw * self.target_std + self.target_mean

    def spread(self, raw):
        """Map a network scale into the head's output space, which carries no shift."""
        return raw if self.binary else raw * self.target_std


@dataclass(frozen=True)
class ArmState:
    """State an arm accumulates along the trajectory.

    ``init`` is given the initial parameters and ``update`` the weights standing
    at the end of each epoch, with the epoch's index counted from the start of
    the fit.
    """

    init: Callable
    update: Callable


@dataclass(frozen=True, eq=False)
class Arm:
    """What an arm adds to the shared hidden blocks.

    ``predict`` is called inside the compiled loop with the fit, the current
    parameters, the padded evaluation rows, a key and the arm's state, and
    returns one entry per name in ``outputs``. Regression locations come back in
    the driver's training space and scales in the same space, since a scale does
    not carry a location's shift.

    ``init_head`` builds the parameters above the hidden blocks, from a key, the
    width beneath them and ``hyper``, and ``loss`` is
    what the fit minimises over them. ``hyper`` names the arm's own columns of
    the config row, which reach it through :attr:`Fit.hyper`. ``metadata`` is what
    the arm needs recorded in ``cell.json`` to reproduce it.

    Arms are compared by identity, which is what keys the program cache: an arm
    built per fit would compile a program per fit, so the ones taking parameters
    come from cached factories.
    """

    outputs: tuple[str, ...]
    predict: Callable
    init_head: Callable
    loss: Callable
    state: ArmState | None = None
    hyper: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


def _objective(task: str) -> str:
    """Locked training loss for a task, leaderboard-aligned."""
    if TASKS[task] == "binary":
        return "logistic"
    return "l1" if primary_metric(task)[0] == "mae" else "l2"


def _transform_features(x: np.ndarray) -> np.ndarray:
    """Compress the Morgan count block; the descriptor block is already normalised."""
    x = np.array(x, dtype=np.float32)
    x[:, :MORGAN_DIM] = np.log1p(x[:, :MORGAN_DIM])
    return x


class _TargetScaler:
    """Centres and scales the training target, and inverts that for predictions.

    Binary targets pass through, since the network's output is mapped by the
    logistic function rather than by a location and a scale.
    """

    def __init__(self, y: np.ndarray, task_type: str):
        self.binary = task_type == "binary"
        self.mean = 0.0 if self.binary else float(np.mean(y))
        spread = 1.0 if self.binary else float(np.std(y))
        # A cluster-grouped inner fold on a small task can be constant.
        self.std = spread if spread > 0 else 1.0

    def forward(self, y: np.ndarray) -> np.ndarray:
        """The target the network is fitted against."""
        return (y - self.mean) / self.std


def dense(key, fan_in: int, fan_out: int, prefix: str) -> dict:
    """One linear layer at PyTorch's default initialisation, named for a checkpoint.

    The names carry no tree definition, so a checkpoint round-trips through a
    flat ``npz``. This takes the second and third keys of ``random.split(key, 3)``
    and a caller building several layers advances along the first.
    """
    _, weight_key, bias_key = random.split(key, 3)
    bound = 1.0 / np.sqrt(fan_in)
    return {
        f"w{prefix}": random.uniform(weight_key, (fan_in, fan_out), minval=-bound, maxval=bound),
        f"b{prefix}": random.uniform(bias_key, (fan_out,), minval=-bound, maxval=bound),
    }


def _init_params(key, n_features: int, n_layers: int, width: int, arm: Arm, hyper: dict) -> dict:
    """The hidden blocks, and whatever the arm puts above them."""
    params = {}
    fan_in = n_features
    for layer in range(n_layers):
        params |= dense(key, fan_in, width, str(layer))
        key = random.split(key, 3)[0]
        fan_in = width
    return params | arm.init_head(key, width, hyper)


def hidden(params: dict, x, n_layers: int, dropout, key=None):
    """Activations of the last hidden block; dropout is applied when ``key`` is given."""
    for layer in range(n_layers):
        x = jnp.maximum(x @ params[f"w{layer}"] + params[f"b{layer}"], 0.0)
        if key is not None:
            key, sub = random.split(key)
            x = x * random.bernoulli(sub, 1.0 - dropout, x.shape) / (1.0 - dropout)
    return x


def readout(params: dict, features):
    """A scalar per row: a logit for binary tasks, a standardized target otherwise."""
    return (features @ params["wout"] + params["bout"]).squeeze(-1)


def forward(params: dict, x, n_layers: int, dropout, key=None):
    """The network's scalar per row, end to end."""
    return readout(params, hidden(params, x, n_layers, dropout, key))


def init_readout(key, width: int, hyper: dict) -> dict:
    """A plain linear layer over the last hidden block."""
    return dense(key, width, 1, "out")


def _loss(objective_name: str, prediction, target, mask):
    """Mean loss over the rows ``mask`` marks as real."""
    if objective_name == "logistic":
        per_row = (
            jnp.maximum(prediction, 0.0)
            - prediction * target
            + jnp.log1p(jnp.exp(-jnp.abs(prediction)))
        )
    elif objective_name == "l1":
        per_row = jnp.abs(prediction - target)
    else:
        per_row = (prediction - target) ** 2
    return (per_row * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def readout_loss(fit: Fit, params: dict, x, y, mask, key):
    """The task's locked objective on the deterministic readout."""
    features = hidden(params, x, fit.n_layers, fit.dropout, key)
    return _loss(fit.objective, readout(params, features), y, mask)


def _predict_deterministic(fit: Fit, params: dict, x, key, state):
    """The network's own output, carrying no spread."""
    return {"mean": fit.location(forward(params, x, fit.n_layers, fit.dropout))}


DETERMINISTIC_ARM = Arm(
    outputs=OUTPUTS,
    predict=_predict_deterministic,
    init_head=init_readout,
    loss=readout_loss,
)


@cache
def _program(
    n_features: int,
    n_layers: int,
    width: int,
    batch_size: int,
    n_train_padded: int,
    n_eval_padded: int,
    n_iterations: int,
    objective_name: str,
    binary: bool,
    arm: Arm = DETERMINISTIC_ARM,
):
    """A jitted sweep for one shape and arm, shared by every fit that matches them.

    Everything a config varies other than its shape is passed in rather than
    closed over, so two configs differing only in dropout, learning rate or an
    arm's own hyperparameters reuse one program.
    """
    n_batches = n_train_padded // batch_size

    def run(
        key,
        x_train,
        y_train,
        valid,
        x_eval,
        dropout,
        learning_rate,
        weight_decay,
        checkpoint_at,
        target_mean,
        target_std,
        hyper,
    ):
        fit = Fit(
            n_layers=n_layers,
            binary=binary,
            objective=objective_name,
            x_train=x_train,
            y_train=y_train,
            valid=valid,
            dropout=dropout,
            target_mean=target_mean,
            target_std=target_std,
            hyper=hyper,
        )
        init_key, epoch_keys = random.split(key)
        optimiser = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        params = _init_params(init_key, n_features, n_layers, width, arm, hyper)
        opt_state = optimiser.init(params)
        arm_state = None if arm.state is None else arm.state.init(params, hyper)

        def epoch(carry, scanned):
            params, opt_state, arm_state, deployed = carry
            epoch_index, epoch_key = scanned
            order_key, step_key = random.split(epoch_key)
            # Permuting the padded length and masking keeps the padded rows out
            # of the loss while every real row is still seen exactly once.
            order = random.permutation(order_key, n_train_padded)

            # The arm's state is carried through the scan untouched, which keeps
            # the compiled program the same shape for an arm that has none: a
            # carry of a different arity fuses differently and moves the last bit
            # of the deterministic arm's predictions.
            def batch(inner, step):
                params, opt_state, arm_state = inner
                taken = lax.dynamic_slice(order, (step * batch_size,), (batch_size,))

                def loss_of(candidate):
                    return arm.loss(
                        fit,
                        candidate,
                        x_train[taken],
                        y_train[taken],
                        valid[taken],
                        random.fold_in(step_key, step),
                    )

                updates, opt_state = optimiser.update(jax.grad(loss_of)(params), opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state, arm_state), None

            (params, opt_state, arm_state), _ = lax.scan(
                batch, (params, opt_state, arm_state), jnp.arange(n_batches)
            )
            if arm.state is not None:
                arm_state = arm.state.update(arm_state, params, epoch_index, hyper)

            # The step after the epoch's last training step: an arm that draws
            # keys takes them from the stream the training steps left off in,
            # rather than from a split that would move the training stream.
            prediction = arm.predict(
                fit, params, x_eval, random.fold_in(step_key, n_batches), arm_state
            )
            # The deployed epoch is known before the fit starts, so it is kept as
            # the scan passes it rather than by stopping there.
            passing = epoch_index + 1 == checkpoint_at
            deployed = jax.tree_util.tree_map(
                lambda fresh, held: jnp.where(passing, fresh, held), params, deployed
            )
            return (params, opt_state, arm_state, deployed), prediction

        (params, _, _, deployed), trajectory = lax.scan(
            epoch,
            (params, opt_state, arm_state, params),
            (jnp.arange(n_iterations), random.split(epoch_keys, n_iterations)),
        )
        return trajectory, params, deployed

    return jax.jit(run)


def _pad(x: np.ndarray, length: int) -> np.ndarray:
    """Extend ``x`` along its first axis with zeros."""
    if len(x) == length:
        return x
    tail = np.zeros((length - len(x), *x.shape[1:]), dtype=x.dtype)
    return np.concatenate([x, tail])


def fit(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    eval_sinks: Sequence[EvalSink],
    *,
    seed: int = 0,
    n_jobs: int = 0,
    n_iterations: int = DEFAULT_ITERATIONS,
    artifacts_dir: Path | None = None,
    checkpoint_at: int | None = None,
    arm: Arm = DETERMINISTIC_ARM,
) -> FitResult:
    """Train for ``n_iterations`` epochs, recording every sink once at the end.

    Args:
        task: TDC ADMET task name.
        x_train: training features.
        y_train: training target, already in the driver's fitting space.
        config: one row of the presampled config list.
        eval_sinks: one per evaluation set; their rows are predicted together and
            the trajectory is split back out by sink and by output.
        seed: seeds the parameter initialisation, the shuffling and the dropout.
        n_jobs: unused; XLA decides its own thread count.
        n_iterations: epoch budget, run in full.
        artifacts_dir: where checkpoints and the head config go.
        checkpoint_at: the epoch that will be deployed, kept as it passes.
        arm: what to make of the weights at the end of an epoch.
    """
    n_layers = int(config["n_layers"])
    width = int(config["hidden_dim"])
    batch_size = int(config["batch_size"])

    scaler = _TargetScaler(y_train, TASKS[task])
    features = _transform_features(x_train)
    labels = np.asarray(scaler.forward(y_train), dtype=np.float32)

    n_train_padded = -(-len(features) // batch_size) * batch_size
    valid = _pad(np.ones(len(features), dtype=np.float32), n_train_padded)

    lengths = [len(sink.features) for sink in eval_sinks]
    stacked = np.concatenate([_transform_features(sink.features) for sink in eval_sinks])
    n_eval_padded = -(-len(stacked) // EVAL_BLOCK) * EVAL_BLOCK

    program = _program(
        features.shape[1],
        n_layers,
        width,
        batch_size,
        n_train_padded,
        n_eval_padded,
        n_iterations,
        _objective(task),
        scaler.binary,
        arm,
    )
    trajectory, last, deployed = program(
        random.PRNGKey(seed),
        jnp.asarray(_pad(features, n_train_padded)),
        jnp.asarray(_pad(labels, n_train_padded)),
        jnp.asarray(valid),
        jnp.asarray(_pad(stacked, n_eval_padded)),
        jnp.float32(config["dropout"]),
        jnp.float32(config["learning_rate"]),
        jnp.float32(config["weight_decay"]),
        jnp.int32(-1 if checkpoint_at is None else checkpoint_at),
        jnp.float32(scaler.mean),
        jnp.float32(scaler.std),
        {name: jnp.asarray(config[name]) for name in arm.hyper},
    )

    recorded = {name: np.asarray(values)[:, : len(stacked)] for name, values in trajectory.items()}
    start = 0
    for sink, length in zip(eval_sinks, lengths, strict=True):
        for name in arm.outputs:
            sink.record(recorded[name][:, start : start + length], output=name)
        start += length

    metadata = {
        "objective": _objective(task),
        "morgan_transform": "log1p",
        "target_mean": scaler.mean,
        "target_std": scaler.std,
        "device": jax.default_backend(),
        # Which command-buffer types XLA wraps kernels in changes the arithmetic,
        # so two runs of one config under different values select different
        # iterations. Recorded whole rather than parsed, since any flag here can
        # move a result.
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "train_rows_padded_to": n_train_padded,
        "eval_rows_padded_to": n_eval_padded,
    } | arm.metadata
    if artifacts_dir is not None:
        _save(deployed, Path(artifacts_dir) / DEPLOYED_CHECKPOINT)
        _save(last, Path(artifacts_dir) / LAST_CHECKPOINT)
        (Path(artifacts_dir) / HEAD_CONFIG_FILE).write_text(
            json.dumps({**config, **metadata, "n_iterations": n_iterations}, indent=2) + "\n"
        )
    return FitResult(n_iterations, metadata)


def _save(params: dict, path: Path) -> None:
    """Write weights atomically to an NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **{name: np.asarray(value) for name, value in params.items()})
    tmp.replace(path)


def load_params(path: str | Path) -> dict:
    """Read a checkpoint back into the parameter dict a fit produced."""
    with np.load(path) as stored:
        return {name: jnp.asarray(stored[name]) for name in stored.files}


def predict(params: dict, x: np.ndarray, config: dict, task: str, metadata: dict) -> np.ndarray:
    """Predictions of a loaded checkpoint, in the head's output space."""
    raw = forward(
        params,
        jnp.asarray(_transform_features(x)),
        int(config["n_layers"]),
        jnp.float32(config["dropout"]),
    )
    if TASKS[task] == "binary":
        return np.asarray(jax.nn.sigmoid(raw))
    return np.asarray(raw * metadata["target_std"] + metadata["target_mean"])


SPEC = Head(
    HEAD=HEAD,
    UQ=UQ,
    DEFAULT_ITERATIONS=DEFAULT_ITERATIONS,
    OUTPUTS=OUTPUTS,
    SCALE_OUTPUTS=SCALE_OUTPUTS,
    POINT_OUTPUT=POINT_OUTPUT,
    CONFIG_LIST=CONFIG_LIST,
    TASK_TYPES=TASK_TYPES,
    fit=fit,
    objectives=objectives,
    ARM=DETERMINISTIC_ARM,
)


def predictive_head(
    uq: str, arm: Arm | Callable[[str], Arm], config_list: str = CONFIG_LIST
) -> Head:
    """The head an arm carrying a predictive distribution defines over this backbone.

    ``arm`` is the arm itself, or a function from task type to it where the two
    task types read the same trained weights through different predictive maps.

    Such an arm records a location and a spread, which gives it two selection
    rules: the leaderboard metric on the location, which is what the
    deterministic head is selected on and keeps the two arms' point predictions
    comparable, and the likelihood, which scores the location and the spread
    together.
    """

    def arm_of(task: str) -> Arm:
        return arm if isinstance(arm, Arm) else arm(TASKS[task])

    def fit_arm(
        task: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        config: dict,
        eval_sinks: Sequence[EvalSink],
        *,
        seed: int = 0,
        n_jobs: int = 0,
        n_iterations: int = DEFAULT_ITERATIONS,
        artifacts_dir: Path | None = None,
        checkpoint_at: int | None = None,
    ) -> FitResult:
        """Train the shared loop, reading each epoch's weights through the arm."""
        return fit(
            task,
            x_train,
            y_train,
            config,
            eval_sinks,
            seed=seed,
            n_jobs=n_jobs,
            n_iterations=n_iterations,
            artifacts_dir=artifacts_dir,
            checkpoint_at=checkpoint_at,
            arm=arm_of(task),
        )

    def objectives_arm(task: str, rows: EvalRows) -> dict[str, Objective]:
        """Select on the leaderboard metric of the task, and on the likelihood."""
        return {
            DEFAULT_SELECTION: leaderboard_objective(task, rows, POINT_OUTPUT),
            "nll": nll_objective(task, rows, POINT_OUTPUT, PREDICTIVE_SCALE),
        }

    return Head(
        HEAD=HEAD,
        UQ=uq,
        DEFAULT_ITERATIONS=DEFAULT_ITERATIONS,
        OUTPUTS=PREDICTIVE_OUTPUTS,
        SCALE_OUTPUTS=(PREDICTIVE_SCALE,),
        POINT_OUTPUT=POINT_OUTPUT,
        CONFIG_LIST=config_list,
        TASK_TYPES=TASK_TYPES,
        fit=fit_arm,
        objectives=objectives_arm,
        ARM=arm,
    )


@cache
def _combined(members: tuple[tuple[str, Arm], ...]) -> Arm:
    """Combine compatible UQ arms in one compiled prediction program."""
    first = members[0][1]
    for uq, member in members:
        if member.init_head is not first.init_head or member.loss is not first.loss:
            raise ValueError(f"{uq}: a shared sweep needs one set of parameters and one loss")
        if member.hyper != first.hyper:
            raise ValueError(f"{uq}: a shared sweep needs one search space")
    stateful = [member for _, member in members if member.state is not None]
    if len(stateful) > 1:
        raise ValueError("a shared sweep carries the state of at most one member")

    def predict(fit_state: Fit, params: dict, x, key, state):
        return {
            f"{uq}{ARM_SEPARATOR}{name}": value
            for uq, member in members
            for name, value in member.predict(fit_state, params, x, key, state).items()
        }

    metadata = {}
    for _, member in members:
        metadata |= member.metadata
    return Arm(
        outputs=tuple(
            f"{uq}{ARM_SEPARATOR}{name}" for uq, member in members for name in member.outputs
        ),
        predict=predict,
        init_head=first.init_head,
        loss=first.loss,
        state=stateful[0].state if stateful else None,
        hyper=first.hyper,
        metadata=metadata | {"shared_sweep": [uq for uq, _ in members]},
    )


def group_fit(members: tuple[tuple[str, Arm | Callable[[str], Arm]], ...]) -> Callable:
    """A fit training one network and recording every member arm off it.

    ``members`` pairs each head's ``uq`` label with its arm, or with a function
    from task type to it. The sinks are :class:`~admet_uq.curves.GroupSink`
    values, which split the recorded ``<uq>:<output>`` names back to one sink per
    arm.
    """

    def fit_group(
        task: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        config: dict,
        eval_sinks: Sequence,
        *,
        seed: int = 0,
        n_jobs: int = 0,
        n_iterations: int = DEFAULT_ITERATIONS,
        artifacts_dir: Path | None = None,
        checkpoint_at: int | None = None,
    ) -> FitResult:
        """Train once, recording each arm's trajectory into its own sink."""
        resolved = tuple(
            (uq, arm if isinstance(arm, Arm) else arm(TASKS[task])) for uq, arm in members
        )
        return fit(
            task,
            x_train,
            y_train,
            config,
            eval_sinks,
            seed=seed,
            n_jobs=n_jobs,
            n_iterations=n_iterations,
            artifacts_dir=artifacts_dir,
            checkpoint_at=checkpoint_at,
            arm=_combined(resolved),
        )

    return fit_group
