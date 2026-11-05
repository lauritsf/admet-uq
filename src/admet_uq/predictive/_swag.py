"""Shared moment accumulation and sampling for checkpoint-resumed SWAG."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from admet_uq.heads.mlp import Fit, forward

N_SAMPLES = 30
RANK = 20


def init(params: dict) -> dict:
    """Initialize moments over the flattened weights."""
    flat, _ = ravel_pytree(params)
    return {
        "mean": jnp.zeros_like(flat),
        "square": jnp.zeros_like(flat),
        "deviations": jnp.zeros((RANK, flat.size), dtype=flat.dtype),
        "count": jnp.zeros((), dtype=jnp.int32),
    }


def update(state: dict, params: dict) -> dict:
    """Fold one parameter snapshot into the moments."""
    flat, _ = ravel_pytree(params)
    count = state["count"] + 1
    weight = 1.0 / count.astype(flat.dtype)
    mean = state["mean"] + (flat - state["mean"]) * weight
    return {
        "mean": mean,
        "square": state["square"] + (flat**2 - state["square"]) * weight,
        "deviations": state["deviations"].at[jnp.mod(count - 1, RANK)].set(flat - mean),
        "count": count,
    }


def _draw(state: dict, unravel, key):
    """Draw one network from the diagonal-plus-low-rank Gaussian."""
    diagonal_key, low_rank_key = random.split(key)
    scale = jnp.sqrt(jnp.clip(state["square"] - state["mean"] ** 2, 0.0, None))
    diagonal = scale * random.normal(diagonal_key, scale.shape, dtype=scale.dtype)
    low_rank = random.normal(low_rank_key, (RANK,), dtype=scale.dtype) @ state["deviations"]
    return unravel(
        state["mean"]
        + diagonal / jnp.sqrt(2.0)
        + low_rank / jnp.sqrt(2.0 * (RANK - 1))
    )


def predict(fit: Fit, params: dict, x, key, state):
    """Return the mean and standard deviation over sampled networks."""
    _, unravel = ravel_pytree(params)

    def one(sample_key):
        drawn = _draw(state, unravel, sample_key)
        return fit.location(forward(drawn, x, fit.n_layers, fit.dropout))

    passes = jax.vmap(one)(random.split(key, N_SAMPLES))
    return {"mean": passes.mean(axis=0), "std": passes.std(axis=0)}
