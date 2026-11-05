"""Shared variational-last-layer equations for post-hoc VBLL."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

N_SAMPLES = 20
DOF = 1.0


def _gaussian_kl(w_mean, w_var, prior_scale):
    """KL from the weight posterior to a zero-mean Gaussian prior."""
    n_outputs, n_features = w_mean.shape
    return 0.5 * (
        (w_mean**2).sum() / prior_scale
        + w_var.sum() / prior_scale
        + (n_outputs * n_features * jnp.log(prior_scale) - jnp.log(w_var).sum())
    )


def _wishart(noise_logdiag, wishart_scale, n_outputs: int):
    """Wishart-prior contribution to the ELBO."""
    dof = (DOF + n_outputs + 1.0) / 2.0
    logdet_precision = -2.0 * noise_logdiag.sum()
    trace_precision = (1.0 / jnp.exp(noise_logdiag) ** 2).sum()
    return dof * logdet_precision - 0.5 * wishart_scale * trace_precision


def _pieces(params: dict):
    """Return the weight mean, weight variance, and noise scale."""
    return params["w_mean"], jnp.exp(params["w_logdiag"]) ** 2, jnp.exp(params["noise_logdiag"])


def init_regression(key, width: int, hyper: dict) -> dict:
    """Initialize a regression last layer."""
    mean_key, logdiag_key, noise_key = random.split(key, 3)
    return {
        "w_mean": random.normal(mean_key, (1, width)),
        "w_logdiag": random.normal(logdiag_key, (1, width)) - 0.5 * np.log(width),
        "noise_logdiag": random.normal(noise_key, (1,)) * jnp.log(hyper["vbll_wishart_scale"]),
    }


def init_classification(key, width: int, hyper: dict) -> dict:
    """Initialize a binary classification last layer."""
    mean_key, logdiag_key, noise_key = random.split(key, 3)
    return {
        "w_mean": random.normal(mean_key, (2, width)) * np.sqrt(2.0 / width),
        "w_logdiag": 1e-3 * random.normal(logdiag_key, (2, width)) - np.log(width),
        "noise_logdiag": random.normal(noise_key, (2,)) - 1.0,
    }


def output_moments(params: dict, features):
    """Return the Gaussian output mean and variance."""
    w_mean, w_var, noise_scale = _pieces(params)
    return features @ w_mean.T, (features**2) @ w_var.T + noise_scale**2


def negative_elbo_regression(params: dict, features, y, mask, hyper: dict, n_train):
    """Negative ELBO for the regression layer."""
    w_mean, w_var, noise_scale = _pieces(params)
    prior_scale = hyper["vbll_prior_scale"] / features.shape[-1]
    noise_precision = 1.0 / noise_scale**2
    mean = (features @ w_mean.T).squeeze(-1)
    log_likelihood = (
        -0.5 * jnp.log(2.0 * jnp.pi * noise_scale**2).sum()
        - 0.5 * ((y - mean) ** 2) * noise_precision.sum()
    )
    trace = 0.5 * ((features**2) @ w_var.T).squeeze(-1) * noise_precision.sum()
    per_row = ((log_likelihood - trace) * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    regularization = 1.0 / jnp.maximum(n_train, 1.0)
    return -(
        per_row
        + regularization
        * (
            _wishart(params["noise_logdiag"], hyper["vbll_wishart_scale"], 1)
            - _gaussian_kl(w_mean, w_var, prior_scale)
        )
    )


def negative_elbo_classification(params: dict, features, y, mask, hyper: dict, n_train):
    """Negative ELBO for the binary classification layer."""
    w_mean, w_var, _ = _pieces(params)
    prior_scale = 2.0 * hyper["vbll_prior_scale"] / features.shape[-1]
    mean, variance = output_moments(params, features)
    linear = mean[:, 1] * y + mean[:, 0] * (1.0 - y)
    bound = linear - jax.nn.logsumexp(mean + 0.5 * variance, axis=-1)
    per_row = (bound * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    regularization = 1.0 / jnp.maximum(n_train, 1.0)
    return -(
        per_row
        + regularization
        * (
            _wishart(params["noise_logdiag"], hyper["vbll_wishart_scale"], 2)
            - _gaussian_kl(w_mean, w_var, prior_scale)
        )
    )
