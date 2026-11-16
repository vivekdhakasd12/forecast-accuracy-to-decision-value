"""Distribution-free calibration of predictive intervals.

Quantile regression gives intervals that are *sharp* but not necessarily
*valid*: a model trained on last summer will happily emit a 90% interval that
covers 74% of this winter.  Conformal prediction fixes the coverage without
assuming anything about the data distribution -- beyond exchangeability, which
is exactly the assumption that concept drift breaks.  Hence two wrappers:

``ConformalizedQuantile``
    Split CQR (Romano, Patterson & Candes, NeurIPS 2019).  Finite-sample
    marginal coverage guarantee, applied *per horizon* (Mondrian conformal) so
    that a 35-hour-ahead interval is not silently subsidised by an easy
    13-hour-ahead one.

``AdaptiveConformal``
    Online adaptive conformal inference (Gibbs & Candes, NeurIPS 2021).  The
    effective miscoverage level is updated after every observation,
    ``alpha_{t+1} = alpha_t + gamma (alpha - err_t)``, which restores long-run
    coverage *under arbitrary distribution shift*.  This is the component that
    connects the forecasting stage to the concept-drift stage of the study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


def _pairs(quantiles: np.ndarray) -> list[tuple[int, int, float]]:
    """Symmetric (lower, upper) index pairs and their nominal miscoverage."""
    q = np.asarray(quantiles, dtype=float)
    out = []
    for i in range(len(q) // 2):
        j = len(q) - 1 - i
        if q[j] <= q[i]:
            continue
        out.append((i, j, float(q[i] + (1.0 - q[j]))))
    return out


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1-alpha) quantile of conformity scores."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


class ConformalizedQuantile:
    """Split-conformal recalibration of an existing quantile forecaster."""

    def __init__(self, model: Forecaster, quantiles, condition_on: str | None = "horizon"):
        self.model = model
        self.quantiles = np.asarray(sorted(float(q) for q in quantiles))
        self.condition_on = condition_on
        self._offsets: dict = {}

    @property
    def name(self) -> str:
        return f"{self.model.name}+cqr"

    def _groups(self, X: pd.DataFrame) -> np.ndarray:
        if self.condition_on is None:
            return np.zeros(len(X), dtype=int)
        return X[self.condition_on].to_numpy()

    def calibrate(self, X_cal: pd.DataFrame, y_cal: np.ndarray) -> "ConformalizedQuantile":
        pred = self.model.predict_quantiles(X_cal, self.quantiles)
        groups = self._groups(X_cal)
        self._offsets = {}
        self._global_offsets = {}
        for lo, hi, alpha in _pairs(self.quantiles):
            # CQR conformity score: how far outside the interval the truth fell.
            score = np.maximum(pred[:, lo] - y_cal, y_cal - pred[:, hi])
            self._global_offsets[(lo, hi)] = conformal_quantile(score, alpha)
            for g in np.unique(groups):
                sel = groups == g
                if sel.sum() >= 30:
                    self._offsets[(lo, hi, g)] = conformal_quantile(score[sel], alpha)
        return self

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> np.ndarray:
        pred = self.model.predict_quantiles(X, self.quantiles).copy()
        groups = self._groups(X)
        for (lo, hi, _alpha) in _pairs(self.quantiles):
            default = self._global_offsets[(lo, hi)]
            off = np.array([self._offsets.get((lo, hi, g), default) for g in groups])
            pred[:, lo] -= off
            pred[:, hi] += off
        pred = np.sort(pred, axis=1)
        pred = self.model._postprocess(X, pred)
        if quantiles is not None:
            want = np.asarray(quantiles, dtype=float)
            # Length must be checked before allclose, which raises on a shape
            # mismatch rather than returning False.
            if len(want) != len(self.quantiles) or not np.allclose(want, self.quantiles):
                pred = np.vstack([np.interp(want, self.quantiles, row) for row in pred])
        return pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class AdaptiveConformal:
    """Online adaptive conformal inference (Gibbs & Candes, NeurIPS 2021).

    The interval is built from a *conformity score*, not by asking the base
    model for ever more extreme quantiles.  That distinction is the whole
    implementation:

        interval_t = [ q_lo(x_t) - Q_t ,  q_hi(x_t) + Q_t ]
        Q_t        = (1 - alpha_t) empirical quantile of recent scores
        alpha_t+1  = alpha_t + gamma * (alpha - err_t)

    A first attempt at this widened the interval by requesting quantiles at
    level ``alpha_t / 2`` from the fitted quantile grid.  Because the grid stops
    at 0.05/0.95 and interpolation clamps outside it, the interval could never
    grow beyond the model's own 90% band no matter how far ``alpha_t`` fell --
    so coverage saturated *below* nominal and adaptation looked harmful.  Going
    through an additive score removes the ceiling: as ``alpha_t`` drops, ``Q_t``
    walks up the empirical score distribution and the interval keeps widening.

    ``gamma = 0`` disables adaptation and recovers ordinary split conformal on
    the same score, which makes the comparison in experiment 3 a clean
    one-variable contrast.
    """

    def __init__(self, model: Forecaster, alpha: float = 0.1, gamma: float = 0.02,
                 window: int = 2000, base_alpha: float = 0.1):
        self.model = model
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.window = int(window)
        self.base_alpha = float(base_alpha)
        self.reset()

    @property
    def name(self) -> str:
        return f"{self.model.name}+aci"

    def reset(self) -> None:
        self.alpha_t = self.alpha
        self._scores: list[float] = []

    def _base_bounds(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        qs = np.array([self.base_alpha / 2.0, 1.0 - self.base_alpha / 2.0])
        pred = self.model.predict_quantiles(X, qs)
        return pred[:, 0], pred[:, 1]

    def seed(self, X_cal: pd.DataFrame, y_cal: np.ndarray) -> "AdaptiveConformal":
        """Prime the score window on held-out calibration data."""
        lo, hi = self._base_bounds(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        self._scores = list(scores[np.isfinite(scores)][-self.window:])
        return self

    def _radius(self) -> float:
        if not self._scores:
            return 0.0
        a = float(np.clip(self.alpha_t, 1e-4, 0.999))
        n = len(self._scores)
        level = min(1.0, np.ceil((n + 1) * (1.0 - a)) / n)
        return float(np.quantile(self._scores, level, method="higher"))

    def run(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Sequentially predict, observe, then update -- never the other order."""
        lo_b, hi_b = self._base_bounds(X)
        n = len(X)
        lo_a = np.empty(n); hi_a = np.empty(n)
        alphas = np.empty(n); miss = np.empty(n)
        for i in range(n):
            r = self._radius()
            lo_i, hi_i = lo_b[i] - r, hi_b[i] + r
            covered = (y[i] >= lo_i) and (y[i] <= hi_i)
            err = 0.0 if covered else 1.0
            lo_a[i], hi_a[i], alphas[i], miss[i] = lo_i, hi_i, self.alpha_t, err
            # Gibbs & Candes update: miss -> lower alpha_t -> wider next interval.
            self.alpha_t = float(np.clip(
                self.alpha_t + self.gamma * (self.alpha - err), 1e-4, 0.999))
            self._scores.append(float(max(lo_b[i] - y[i], y[i] - hi_b[i])))
            if len(self._scores) > self.window:
                self._scores.pop(0)
        return pd.DataFrame({"lower": lo_a, "upper": hi_a, "alpha_t": alphas,
                             "miss": miss, "y": y}, index=X.index)
