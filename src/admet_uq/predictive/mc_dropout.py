"""MC dropout: the spread over stochastic forward passes.

Dropout stays on at evaluation, so K passes over the same rows are K draws from
the predictive distribution the trained network implies. The masks are
exchangeable draws, so the passes are taken as one vmapped call rather than a
loop over K.

Each pass is mapped into the head's output space before the passes are
summarised, so a binary task's spread is over probabilities and a regression
task's is in the driver's training space.
"""

from __future__ import annotations

import jax
from jax import random

from admet_uq.heads.mlp import (
    PREDICTIVE_OUTPUTS,
    Arm,
    Fit,
    forward,
    init_readout,
    predictive_head,
    readout_loss,
)

#: Stochastic passes per evaluation. Fixed rather than searched: K resolves the
#: same predictive distribution more finely rather than choosing a different one.
N_PASSES = 30


def _predict(fit: Fit, params: dict, x, key, state):
    """Mean and standard deviation over ``N_PASSES`` passes."""

    def one_pass(pass_key):
        return fit.location(forward(params, x, fit.n_layers, fit.dropout, pass_key))

    passes = jax.vmap(one_pass)(random.split(key, N_PASSES))
    return {"mean": passes.mean(axis=0), "std": passes.std(axis=0)}


ARM = Arm(
    outputs=PREDICTIVE_OUTPUTS,
    predict=_predict,
    init_head=init_readout,
    loss=readout_loss,
    metadata={"mc_dropout_passes": N_PASSES},
)

SPEC = predictive_head("mcd", ARM)
