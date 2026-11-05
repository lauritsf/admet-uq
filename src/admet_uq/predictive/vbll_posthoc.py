"""Variational Bayesian last layer fitted to frozen MLP features.

Prior and Wishart scales are selected from fixed grids by fitted ELBO.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from admet_uq.predictive._vbll import (
    N_SAMPLES,
    init_classification,
    init_regression,
    negative_elbo_classification,
    negative_elbo_regression,
    output_moments,
)

#: Prior scale on the last layer's weights, over the range the trained arm searches.
PRIOR_GRID = tuple(float(v) for v in np.logspace(-2.0, 2.0, 9))

#: Wishart scale on the noise precision, over the range the trained arm searches.
WISHART_GRID = tuple(float(v) for v in np.logspace(-4.0, 0.0, 9))

#: Adam steps per grid point, run in full.
STEPS = 1000

#: Step size for those, on a problem whose scale the frozen features fix.
LEARNING_RATE = 1e-2


def _elbo(binary: bool):
    """The negative ELBO for a task type, over features rather than over rows."""
    return negative_elbo_classification if binary else negative_elbo_regression


def _fit_one(features, y, binary: bool, hyper: dict, key):
    """Fit the layer at one grid point, returning its parameters and final ELBO."""
    width = features.shape[-1]
    init = init_classification if binary else init_regression
    params = init(key, width, hyper)
    mask = jnp.ones(features.shape[0], dtype=features.dtype)
    n_train = mask.sum()
    loss = _elbo(binary)

    optimiser = optax.adam(LEARNING_RATE)
    state = optimiser.init(params)

    def step(carry, _):
        params, state = carry
        value, grads = jax.value_and_grad(loss)(params, features, y, mask, hyper, n_train)
        updates, state = optimiser.update(grads, state, params)
        return (optax.apply_updates(params, updates), state), value

    (params, _), values = jax.lax.scan(step, (params, state), None, length=STEPS)
    return params, values[-1]


_fit_one_jit = jax.jit(_fit_one, static_argnums=(2,))


def fit_layer(features: np.ndarray, y: np.ndarray, task_type: str, seed: int = 0) -> dict:
    """Fit the layer at every grid point and keep the one with the largest ELBO.

    ``features`` are the frozen activations of the training rows and ``y`` the
    target the network was trained against, standardized for regression and the
    label itself for a binary task.
    """
    binary = task_type == "binary"
    features = jnp.asarray(features, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)

    best: dict | None = None
    for prior_scale in PRIOR_GRID:
        for wishart_scale in WISHART_GRID:
            hyper = {
                "vbll_prior_scale": jnp.float32(prior_scale),
                "vbll_wishart_scale": jnp.float32(wishart_scale),
            }
            params, value = _fit_one_jit(features, y, binary, hyper, random.PRNGKey(seed))
            value = float(value)
            if not np.isfinite(value):
                continue
            if best is None or value < best["negative_elbo"]:
                best = {
                    "params": params,
                    "negative_elbo": value,
                    "vbll_prior_scale": prior_scale,
                    "vbll_wishart_scale": wishart_scale,
                }
    if best is None:
        raise ValueError("every grid point gave a non-finite ELBO")
    return best


def predict(
    params: dict,
    features: np.ndarray,
    task_type: str,
    *,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """The predictive location and spread at ``features``, in the task's units.

    The spread carries no location, so the target scale multiplies it where the
    mean also takes the shift.
    """
    features = jnp.asarray(features, dtype=jnp.float32)
    mean, variance = output_moments(params, features)
    if task_type == "binary":
        draws = mean + jnp.sqrt(variance) * random.normal(
            random.PRNGKey(seed), (N_SAMPLES, *mean.shape), dtype=mean.dtype
        )
        positive = jax.nn.softmax(draws, axis=-1)[..., 1]
        return {
            "mean": np.asarray(positive.mean(axis=0)),
            "std": np.asarray(positive.std(axis=0)),
        }
    return {
        "mean": np.asarray(mean.squeeze(-1) * target_std + target_mean),
        "std": np.asarray(jnp.sqrt(variance.squeeze(-1)) * target_std),
    }
