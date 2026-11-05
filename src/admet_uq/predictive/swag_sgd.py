"""SWAG collected after a selected MLP checkpoint under a dedicated SGD phase."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax, random
from jax.flatten_util import ravel_pytree

from admet_uq.data import TASKS
from admet_uq.heads.mlp import (
    EVAL_BLOCK,
    Fit,
    _objective,
    _pad,
    _TargetScaler,
    _transform_features,
    forward,
    readout_loss,
)
from admet_uq.predictive import _swag

UQ = "swag_sgd"
MOMENTUM = 0.9
REFERENCE_BATCH_SIZE = 128
DEFAULT_SWA_LR_AT_BATCH_128 = 0.005
DEFAULT_SETTLING_EPOCHS = 160
DEFAULT_COLLECTION_EPOCHS = 140
DEFAULT_SNAPSHOTS_PER_EPOCH = 1


@dataclass(frozen=True)
class CollectionResult:
    """Predictive summary and diagnostics from one checkpoint continuation."""

    mean: np.ndarray
    std: np.ndarray
    diagnostics: dict[str, float | int]


@cache
def _program(
    n_features: int,
    n_layers: int,
    batch_size: int,
    n_train_padded: int,
    n_eval_padded: int,
    objective_name: str,
    binary: bool,
    settling_epochs: int,
    collection_epochs: int,
    snapshots_per_epoch: int,
):
    """Compile one checkpoint continuation for a padded train/evaluation shape."""
    n_batches = n_train_padded // batch_size
    total_epochs = settling_epochs + collection_epochs
    snapshots_this_epoch = min(snapshots_per_epoch, n_batches)

    def run(
        params,
        key,
        x_train,
        y_train,
        valid,
        x_eval,
        dropout,
        swa_lr,
        weight_decay,
        target_mean,
        target_std,
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
            hyper={},
        )
        train_key, predictive_key = random.split(key)
        optimiser = optax.sgd(learning_rate=swa_lr, momentum=MOMENTUM)
        opt_state = optimiser.init(params)
        swag_state = _swag.init(params)
        initial = params

        def epoch(carry, scanned):
            params, opt_state, swag_state = carry
            epoch_index, epoch_key = scanned
            order_key, step_key = random.split(epoch_key)
            order = random.permutation(order_key, n_train_padded)

            def batch(inner, step):
                params, opt_state, swag_state = inner
                taken = lax.dynamic_slice(order, (step * batch_size,), (batch_size,))

                def loss_of(candidate):
                    return readout_loss(
                        fit,
                        candidate,
                        x_train[taken],
                        y_train[taken],
                        valid[taken],
                        random.fold_in(step_key, step),
                    )

                gradients = jax.grad(loss_of)(params)
                # PyTorch's SGD ``weight_decay`` adds an L2 term to the gradient;
                # do the same rather than carrying AdamW's decoupled update into
                # the collection phase.
                gradients = jax.tree_util.tree_map(
                    lambda gradient, parameter: gradient + weight_decay * parameter,
                    gradients,
                    params,
                )
                updates, opt_state = optimiser.update(gradients, opt_state, params)
                params = optax.apply_updates(params, updates)

                # Divide the epoch into equal step intervals.  When an epoch
                # has fewer steps than requested snapshots, every distinct
                # post-update iterate is collected and no parameter vector is
                # duplicated.
                before = step * snapshots_this_epoch // n_batches
                after = (step + 1) * snapshots_this_epoch // n_batches
                should_collect = (epoch_index >= settling_epochs) & (after > before)
                swag_state = lax.cond(
                    should_collect,
                    lambda state: _swag.update(state, params),
                    lambda state: state,
                    swag_state,
                )
                return (params, opt_state, swag_state), None

            (params, opt_state, swag_state), _ = lax.scan(
                batch, (params, opt_state, swag_state), jnp.arange(n_batches)
            )
            return (params, opt_state, swag_state), None

        (params, _, swag_state), _ = lax.scan(
            epoch,
            (params, opt_state, swag_state),
            (jnp.arange(total_epochs), random.split(train_key, total_epochs)),
        )
        predictive = _swag.predict(fit, params, x_eval, predictive_key, swag_state)

        def objective_loss(candidate):
            raw = forward(candidate, x_train, n_layers, dropout)
            if objective_name == "logistic":
                per_row = (
                    jnp.maximum(raw, 0.0)
                    - raw * y_train
                    + jnp.log1p(jnp.exp(-jnp.abs(raw)))
                )
            elif objective_name == "l1":
                per_row = jnp.abs(raw - y_train)
            else:
                per_row = (raw - y_train) ** 2
            return (per_row * valid).sum() / jnp.maximum(valid.sum(), 1.0)

        initial_flat, unravel = ravel_pytree(initial)
        final_flat, _ = ravel_pytree(params)
        mean_params = unravel(swag_state["mean"])
        diagonal_std = jnp.sqrt(
            jnp.clip(swag_state["square"] - swag_state["mean"] ** 2, 0.0, None)
        )
        diagnostics = {
            "initial_train_loss": objective_loss(initial),
            "final_train_loss": objective_loss(params),
            "swag_mean_train_loss": objective_loss(mean_params),
            "parameter_displacement_l2": jnp.linalg.norm(final_flat - initial_flat),
            "parameter_l2": jnp.linalg.norm(initial_flat),
            "mean_diagonal_std": diagonal_std.mean(),
            "max_diagonal_std": diagonal_std.max(),
            "n_snapshots": swag_state["count"],
        }
        return predictive, diagnostics

    return jax.jit(run)


def collect(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    params: dict,
    config: dict,
    *,
    seed: int,
    swa_lr_at_batch_128: float = DEFAULT_SWA_LR_AT_BATCH_128,
    settling_epochs: int = DEFAULT_SETTLING_EPOCHS,
    collection_epochs: int = DEFAULT_COLLECTION_EPOCHS,
    snapshots_per_epoch: int = DEFAULT_SNAPSHOTS_PER_EPOCH,
) -> CollectionResult:
    """Continue one selected checkpoint and return its SWAG predictive.

    ``y_train`` is already in the driver's fitting space.  The target scaler is
    reconstructed from exactly the source refit's rows and checked against the
    metadata stored beside the checkpoint before any update is made.
    """
    if settling_epochs < 0:
        raise ValueError("settling_epochs must be non-negative")
    if collection_epochs < 2:
        raise ValueError("collection_epochs must be at least two")
    if snapshots_per_epoch < 1:
        raise ValueError("snapshots_per_epoch must be positive")
    if swa_lr_at_batch_128 <= 0:
        raise ValueError("swa_lr_at_batch_128 must be positive")

    scaler = _TargetScaler(y_train, TASKS[task])
    for name, expected, found in (
        ("target_mean", scaler.mean, float(config["target_mean"])),
        ("target_std", scaler.std, float(config["target_std"])),
    ):
        if not np.isclose(expected, found, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"source checkpoint {name}={found} disagrees with fit rows ({expected})"
            )

    features = _transform_features(x_train)
    labels = np.asarray(scaler.forward(y_train), dtype=np.float32)
    batch_size = int(config["batch_size"])
    # The source HPO varies batch size from 16 to 256.  A fixed per-step SGD
    # rate would therefore move a batch-16 model eight times as far per epoch as
    # the batch-128 convention at which the collection rate is declared.
    swa_lr = swa_lr_at_batch_128 * batch_size / REFERENCE_BATCH_SIZE
    n_train_padded = -(-len(features) // batch_size) * batch_size
    valid = _pad(np.ones(len(features), dtype=np.float32), n_train_padded)
    evaluation = _transform_features(x_eval)
    n_eval_padded = -(-len(evaluation) // EVAL_BLOCK) * EVAL_BLOCK

    program = _program(
        features.shape[1],
        int(config["n_layers"]),
        batch_size,
        n_train_padded,
        n_eval_padded,
        _objective(task),
        TASKS[task] == "binary",
        settling_epochs,
        collection_epochs,
        snapshots_per_epoch,
    )
    predictive, diagnostics = program(
        params,
        random.PRNGKey(seed),
        jnp.asarray(_pad(features, n_train_padded)),
        jnp.asarray(_pad(labels, n_train_padded)),
        jnp.asarray(valid),
        jnp.asarray(_pad(evaluation, n_eval_padded)),
        jnp.float32(config["dropout"]),
        jnp.float32(swa_lr),
        jnp.float32(config["weight_decay"]),
        jnp.float32(scaler.mean),
        jnp.float32(scaler.std),
    )
    mean = np.asarray(predictive["mean"][: len(evaluation)])
    std = np.asarray(predictive["std"][: len(evaluation)])
    checked = {
        name: int(value) if name == "n_snapshots" else float(value)
        for name, value in diagnostics.items()
    }
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise FloatingPointError("checkpoint SWAG produced non-finite predictions")
    if not all(np.isfinite(value) for value in checked.values()):
        raise FloatingPointError("checkpoint SWAG produced non-finite diagnostics")
    expected_snapshots = collection_epochs * min(snapshots_per_epoch, n_train_padded // batch_size)
    if checked["n_snapshots"] != expected_snapshots:
        raise RuntimeError(
            f"collected {checked['n_snapshots']} snapshots, expected {expected_snapshots}"
        )
    return CollectionResult(mean=mean, std=std, diagnostics=checked)
