"""Last-layer Laplace approximation for a trained MLP.

Hidden features are fixed, and prior precision is selected by marginal
likelihood over :data:`PRIOR_GRID`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from admet_uq.heads.mlp import (
    PREDICTIVE_OUTPUTS,
    Arm,
    Fit,
    hidden,
    init_readout,
    predictive_head,
    readout,
    readout_loss,
)

#: Prior precisions the marginal likelihood is compared over.
PRIOR_GRID = jnp.logspace(-4.0, 4.0, 33)

#: Smallest noise variance a regression fit is credited with, in the standardized
#: target's own units.
MIN_NOISE = 1e-6


def _activations(params: dict, fit: Fit, x):
    """The readout's inputs, and the logit or standardized target it maps them to."""
    features = hidden(params, x, fit.n_layers, fit.dropout)
    return features, readout(params, features)


def _prior_precision(eigenvalues, weights_norm, n_parameters: int):
    """The grid's argmax of the Laplace log marginal likelihood.

    The fit term is the same at every candidate, since the weights are already
    trained, so what is compared is the prior's penalty on them against the
    volume its precision leaves.
    """
    log_marginal = 0.5 * (
        n_parameters * jnp.log(PRIOR_GRID)
        - jnp.log(eigenvalues[None, :] + PRIOR_GRID[:, None]).sum(axis=1)
        - PRIOR_GRID * weights_norm
    )
    return PRIOR_GRID[jnp.argmax(log_marginal)]


def _with_bias(features):
    """Append the constant column the readout's bias multiplies."""
    return jnp.concatenate([features, jnp.ones((features.shape[0], 1))], axis=1)


def _posterior(params: dict, fit: Fit, train_features, curvature):
    """Eigenbasis of the last layer's posterior precision, and its eigenvalues."""
    design = _with_bias(train_features)
    eigenvalues, basis = jnp.linalg.eigh(design.T @ (design * (curvature * fit.valid)[:, None]))

    weights = jnp.concatenate([params["wout"].squeeze(-1), params["bout"]])
    prior = _prior_precision(jnp.clip(eigenvalues, 0.0, None), (weights**2).sum(), weights.size)
    return basis, jnp.clip(eigenvalues, 0.0, None) + prior


def _variance(basis, precision, features):
    """The quadratic form of each row against the inverse posterior precision."""
    return ((_with_bias(features) @ basis) ** 2 / precision).sum(axis=1)


def _predict_regression(fit: Fit, params: dict, x, key, state):
    """A Gaussian whose variance is the weight uncertainty plus the fitted noise."""
    train_features, train_prediction = _activations(params, fit, fit.x_train)
    residual = train_prediction - fit.y_train
    noise = jnp.maximum(
        (residual**2 * fit.valid).sum() / jnp.maximum(fit.valid.sum(), 1.0), MIN_NOISE
    )
    basis, precision = _posterior(params, fit, train_features, 1.0 / noise)

    features, prediction = _activations(params, fit, x)
    variance = _variance(basis, precision, features) + noise
    return {"mean": fit.location(prediction), "std": fit.spread(jnp.sqrt(variance))}


def _predict_classification(fit: Fit, params: dict, x, key, state):
    """A probability under the probit approximation, and the spread it implies."""
    train_features, train_logit = _activations(params, fit, fit.x_train)
    train_probability = jax.nn.sigmoid(train_logit)
    basis, precision = _posterior(
        params, fit, train_features, train_probability * (1.0 - train_probability)
    )

    features, logit = _activations(params, fit, x)
    variance = _variance(basis, precision, features)
    probability = jax.nn.sigmoid(logit / jnp.sqrt(1.0 + jnp.pi * variance / 8.0))
    # The delta method through the logistic function: the spread of the
    # probability the same Gaussian over logits implies.
    return {"mean": probability, "std": probability * (1.0 - probability) * jnp.sqrt(variance)}


def _arm(binary: bool) -> Arm:
    """The posterior for a task type, over the arm the deterministic head trains."""
    return Arm(
        outputs=PREDICTIVE_OUTPUTS,
        predict=_predict_classification if binary else _predict_regression,
        init_head=init_readout,
        loss=readout_loss,
        metadata={
            "laplace_prior_grid": [float(PRIOR_GRID[0]), float(PRIOR_GRID[-1]), PRIOR_GRID.size],
            "laplace_predictive": "probit" if binary else "gaussian",
        },
    )


BINARY_ARM = _arm(True)
REGRESSION_ARM = _arm(False)


def arm(task_type: str) -> Arm:
    """The arm for a task type."""
    return BINARY_ARM if task_type == "binary" else REGRESSION_ARM


SPEC = predictive_head("laplace", arm)
