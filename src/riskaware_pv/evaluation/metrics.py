"""Scoring rules for point and probabilistic forecasts.

Deterministic accuracy alone cannot rank probabilistic forecasts, and coverage
alone cannot either -- an interval of [0, capacity] has perfect coverage and
zero value.  The study therefore reports a *proper* scoring rule (CRPS, via its
quantile decomposition) alongside the calibration/sharpness pair (PICP, PINAW)
that operators actually read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------- #
# point forecasts
# --------------------------------------------------------------------------- #
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(yhat - y))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + EPS))


def skill_score(score: float, reference: float) -> float:
    """Fractional improvement over a reference score (1 = perfect, 0 = no skill)."""
    return float(1.0 - score / (reference + EPS))


# --------------------------------------------------------------------------- #
# probabilistic forecasts
# --------------------------------------------------------------------------- #
def pinball_loss(y: np.ndarray, q_pred: np.ndarray, quantiles) -> float:
    """Mean pinball (quantile) loss averaged over samples and quantile levels."""
    q = np.asarray(quantiles, dtype=float)[None, :]
    err = y[:, None] - q_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def pinball_by_quantile(y: np.ndarray, q_pred: np.ndarray, quantiles) -> pd.Series:
    q = np.asarray(quantiles, dtype=float)
    err = y[:, None] - q_pred
    loss = np.maximum(q[None, :] * err, (q[None, :] - 1.0) * err).mean(axis=0)
    return pd.Series(loss, index=q, name="pinball")


def crps_from_quantiles(y: np.ndarray, q_pred: np.ndarray, quantiles) -> float:
    """CRPS approximated from a quantile grid.

    Uses the identity CRPS(F, y) = 2 * integral_0^1 pinball_q(F, y) dq, evaluated
    by the trapezoid rule on the available grid.  With a symmetric grid such as
    (0.05 ... 0.95) this is a slight under-estimate of the true CRPS because the
    extreme tails are unrepresented -- it is consistent across models, which is
    what matters for ranking, and we report it as such.
    """
    q = np.asarray(quantiles, dtype=float)
    order = np.argsort(q)
    q, q_pred = q[order], q_pred[:, order]
    err = y[:, None] - q_pred
    pin = np.maximum(q[None, :] * err, (q[None, :] - 1.0) * err)
    # np.trapezoid is NumPy 2.0+; np.trapz is the older spelling. Colab and other
    # hosted environments are not always on the same major version as the dev box.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(2.0 * integrate(pin, x=q, axis=1).mean())


def picp(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability -- the empirical coverage."""
    return float(np.mean((y >= lower) & (y <= upper)))


def pinaw(lower: np.ndarray, upper: np.ndarray, scale: float | None = None) -> float:
    """Prediction Interval Normalised Average Width (sharpness; lower is better)."""
    width = np.mean(upper - lower)
    denom = scale if scale is not None else 1.0
    return float(width / (denom + EPS))


def ace(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float) -> float:
    """Average Coverage Error: realised minus nominal coverage (0 is ideal)."""
    return float(picp(y, lower, upper) - nominal)


def winkler_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  alpha: float) -> float:
    """Interval score: width plus a penalty proportional to the miss distance.

    Unlike PICP it cannot be gamed by widening the interval, so it is the right
    single number when comparing calibrated against uncalibrated models.
    """
    width = upper - lower
    below = 2.0 / alpha * (lower - y) * (y < lower)
    above = 2.0 / alpha * (y - upper) * (y > upper)
    return float(np.mean(width + below + above))


def reliability_curve(y: np.ndarray, q_pred: np.ndarray, quantiles) -> pd.DataFrame:
    """Nominal versus realised exceedance probability for each quantile level."""
    q = np.asarray(quantiles, dtype=float)
    observed = [(y <= q_pred[:, i]).mean() for i in range(len(q))]
    return pd.DataFrame({"nominal": q, "observed": observed,
                         "deviation": np.array(observed) - q})


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def evaluate_all(y: np.ndarray, point: np.ndarray, q_pred: np.ndarray | None,
                 quantiles=None, scale: float = 1.0,
                 reference_mae: float | None = None) -> dict:
    """One row of the results table for a single model on a single test slice."""
    out = {
        "mae": mae(y, point),
        "rmse": rmse(y, point),
        "bias": bias(y, point),
        "r2": r2(y, point),
        "nmae_pct": 100.0 * mae(y, point) / (scale + EPS),
    }
    if reference_mae is not None:
        out["mae_skill_vs_persistence"] = skill_score(out["mae"], reference_mae)
    if q_pred is not None and quantiles is not None:
        q = np.asarray(quantiles, dtype=float)
        out["pinball"] = pinball_loss(y, q_pred, q)
        out["crps"] = crps_from_quantiles(y, q_pred, q)
        for lo_q, hi_q in ((0.05, 0.95), (0.1, 0.9), (0.25, 0.75)):
            if lo_q in set(np.round(q, 4)) and hi_q in set(np.round(q, 4)):
                i = int(np.argmin(np.abs(q - lo_q)))
                j = int(np.argmin(np.abs(q - hi_q)))
                nominal = round(hi_q - lo_q, 6)
                tag = f"{int(round(nominal * 100))}"
                lo_p, hi_p = q_pred[:, i], q_pred[:, j]
                out[f"picp{tag}"] = picp(y, lo_p, hi_p)
                out[f"pinaw{tag}"] = pinaw(lo_p, hi_p, scale)
                out[f"ace{tag}"] = ace(y, lo_p, hi_p, nominal)
                out[f"winkler{tag}"] = winkler_score(y, lo_p, hi_p, 1.0 - nominal)
    return out
