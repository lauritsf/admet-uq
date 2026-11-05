"""Post-hoc calibration fitted on each cell's calibration split.

Binary predictions use temperature scaling. Regression point predictions use
a constant Gaussian scale; predictive scales can be multiplied or augmented
with a noise floor. Log-target tasks are calibrated in log space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar, shgo
from scipy.special import expit, logit

from admet_uq.metrics import MIN_SCALE

CLASSIFICATION_CALIBRATORS = ("none", "temperature")
REGRESSION_CALIBRATORS = ("gaussian",)
DISTRIBUTIONAL_CALIBRATORS = ("none", "scaled", "noise_floor")

_EPS = 1e-7


@dataclass(frozen=True)
class Calibrated:
    """Calibrated predictions and fitted calibration parameters."""

    y_pred: np.ndarray
    pred_std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    in_set: np.ndarray | None = None
    set_size: np.ndarray | None = None
    scale: float = float("nan")
    offset: float = float("nan")
    fallback: str = ""


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    """Logit with clipping, so a saturated stored probability stays finite."""
    return logit(np.clip(np.asarray(probability, dtype=float), _EPS, 1 - _EPS))


# Bound on ``|log T|``. At this temperature every probability is within 1e-2 of
# a half, so the fit has nowhere further to go that a metric can tell apart.
TEMPERATURE_LOG_BOUND = 4.0

# How close to that bound counts as resting on it: bounded Brent stops within
# its xatol of a boundary optimum, so an exact test never fires.
_AT_BOUND = 1e-4

SEPARATED_FALLBACK = "separated_calibration_slice"

TEMPERATURE_BOUND_FALLBACK = "temperature_at_bound"


def separates_at_half(calib_prob: np.ndarray, calib_y: np.ndarray) -> bool:
    """Whether ``p = 0.5`` separates the slice, making the temperature fit ill-posed.

    Temperature scaling has no intercept, so its boundary is pinned at
    ``logit(p) = 0``. Sharpening then drives every correctly sided point's loss to
    zero and every wrongly sided one's to infinity, so the likelihood runs away
    when no point is wrongly sided.

    The test is non-strict because a point at exactly 0.5 contributes a constant
    ``log 2`` at every temperature and does not arrest the runaway either
    (quasi-complete separation, Albert & Anderson 1984).
    """
    prob = np.asarray(calib_prob, dtype=float)
    positive = np.asarray(calib_y, dtype=float) > 0.5
    return bool(np.all(prob[positive] >= 0.5) and np.all(prob[~positive] <= 0.5))


def fit_temperature(calib_prob: np.ndarray, calib_y: np.ndarray) -> float:
    """Fit a single temperature by minimising calibration NLL.

    Applied in logit space, ``sigmoid(logit(p) / T)``, so it is strictly
    monotone and leaves ranking metrics untouched. ``T > 1`` softens an
    over-confident model, ``T < 1`` sharpens an under-confident one.
    """
    z = _safe_logit(calib_prob)
    y = np.asarray(calib_y, dtype=float)

    def negative_log_likelihood(log_temperature: float) -> float:
        p = np.clip(expit(z / np.exp(log_temperature)), _EPS, 1 - _EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # Optimise log T so the parameter is unconstrained in sign and T stays positive.
    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(-TEMPERATURE_LOG_BOUND, TEMPERATURE_LOG_BOUND),
        method="bounded",
    )
    return float(np.exp(result.x))


def calibrate_classification(
    calibrator: str, calib_prob: np.ndarray, calib_y: np.ndarray, test_prob: np.ndarray
) -> Calibrated:
    """Apply one classification calibrator, fitted on the calibration slice.

    Temperature falls back to ``T = 1`` on a separated slice: a slice with no
    examples near the decision boundary carries no evidence about confidence
    there. See :func:`separates_at_half`.

    Temperature divides the logit and cannot shift it, so a slice whose base rate
    sits outside the range the probabilities span is fitted best by the flattest
    model the search allows, and the likelihood keeps improving to ``T`` of
    infinity. That fit stops at :data:`TEMPERATURE_LOG_BOUND` and reports
    :data:`TEMPERATURE_BOUND_FALLBACK`, where a family carrying an intercept has
    somewhere else to go.
    """
    test_prob = np.asarray(test_prob, dtype=float)
    if calibrator == "none":
        return Calibrated(y_pred=test_prob)
    if calibrator == "temperature":
        if separates_at_half(calib_prob, calib_y):
            return Calibrated(y_pred=test_prob, scale=1.0, fallback=SEPARATED_FALLBACK)
        temperature = fit_temperature(calib_prob, calib_y)
        return Calibrated(
            y_pred=expit(_safe_logit(test_prob) / temperature),
            scale=temperature,
            fallback=(
                TEMPERATURE_BOUND_FALLBACK
                if abs(np.log(temperature)) >= TEMPERATURE_LOG_BOUND - _AT_BOUND
                else ""
            ),
        )
    raise ValueError(f"unknown classification calibrator {calibrator!r}")


def fit_gaussian_scale(calib_pred: np.ndarray, calib_y: np.ndarray) -> float:
    """Maximum-likelihood constant predictive scale from calibration residuals."""
    residual = np.asarray(calib_y, dtype=float) - np.asarray(calib_pred, dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


def fit_scale_factor(calib_pred: np.ndarray, calib_y: np.ndarray, calib_scale: np.ndarray) -> float:
    """Maximum-likelihood multiplier on a head's own predictive scale.

    Under a Gaussian of scale ``factor * scale_i`` the likelihood is maximised at
    the root mean square of the residuals measured in their own predicted scales,
    so the fit is closed-form. A factor above one widens a distribution the head
    made too sharp.
    """
    scale = np.maximum(np.asarray(calib_scale, dtype=float), MIN_SCALE)
    residual = np.asarray(calib_y, dtype=float) - np.asarray(calib_pred, dtype=float)
    return float(np.sqrt(np.mean((residual / scale) ** 2)))


def fit_noise_floor(calib_pred: np.ndarray, calib_y: np.ndarray, calib_scale: np.ndarray) -> float:
    """Maximum-likelihood noise scale added in quadrature to a head's own.

    Under a Gaussian of variance ``scale_i^2 + noise^2`` the likelihood has no
    closed form and can have more than one local minimum, so the noise variance
    is found by a deterministic global scalar search in log-variance. No
    optimum can lie above the largest squared residual: from there onwards every
    summand's derivative is non-negative. The mean squared residual is only the
    solution when the supplied scale is constant at zero, and is not a valid
    bound when the scale varies.

    An arm whose spread is the variation over stochastic passes or sampled
    networks measures how much the model moves, not how much the measurement
    does. This supplies the second term; :func:`fit_scale_factor` cannot, a
    multiplier on a spread that is small everywhere staying small everywhere.
    """
    scale = np.maximum(np.asarray(calib_scale, dtype=float), MIN_SCALE)
    residual = np.asarray(calib_y, dtype=float) - np.asarray(calib_pred, dtype=float)
    upper = float(np.max(residual**2))
    if upper <= 0.0:
        return 0.0

    def negative_log_likelihood(variance: float) -> float:
        total = scale**2 + variance
        return float(np.sum(np.log(total) + residual**2 / total))

    # A single Brent search is not valid here: heterogeneous supplied scales can
    # make the likelihood multimodal. Search positive variance globally on the
    # numerically resolvable log scale, then compare the excluded zero endpoint.
    positive_residual = residual[residual != 0.0] ** 2
    smallest_relevant = min(float(np.min(scale**2)), float(np.min(positive_residual)))
    lower = max(smallest_relevant * np.finfo(float).eps, np.finfo(float).tiny)

    def log_variance_objective(log_variance: np.ndarray) -> float:
        return negative_log_likelihood(float(np.exp(log_variance[0])))

    result = shgo(
        log_variance_objective,
        [(float(np.log(lower)), float(np.log(upper)))],
        n=128,
        iters=1,
        sampling_method="simplicial",
    )
    variance = float(np.exp(result.x[0]))
    if negative_log_likelihood(0.0) <= negative_log_likelihood(variance):
        variance = 0.0
    return float(np.sqrt(variance))
