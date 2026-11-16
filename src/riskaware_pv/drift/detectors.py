"""Concept-drift detection on the forecast error stream.

Weather regimes shift, panels soil and degrade, inverters get replaced, and
market rules change.  A model trained on 2018 is not obviously the right model
for 2020, but retraining is not free either -- it costs compute, it invalidates
audit trails, and on a short history it can make things *worse* by discarding
seasons the model still needs.

The question this module supports is therefore not "is there drift?" but "does
*detecting* drift and retraining on the trigger beat simply retraining on a
fixed schedule, at the same retraining budget?"  That is a falsifiable claim,
and the answer in the results is not the obvious one.

Detectors implemented
---------------------
ADWIN            adaptive windowing on the absolute-error stream (Bifet & Gavalda 2007)
Page-Hinkley     cumulative-deviation test, classic for gradual mean shift
PSI              population stability index on the *feature* distribution
KS               two-sample Kolmogorov-Smirnov on a reference vs recent window
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

try:  # pragma: no cover
    from river.drift import ADWIN as _RiverADWIN
    _HAS_RIVER = True
except Exception:  # pragma: no cover
    _HAS_RIVER = False


@dataclass
class DriftEvent:
    index: int
    timestamp: pd.Timestamp | None
    detector: str
    statistic: float


class ReferenceStandardizer:
    """Freeze the mean/std of a warm-up window and z-score everything after it.

    Both ADWIN and Page-Hinkley threshold *absolute* deviations, so their
    default parameters are implicitly calibrated to a signal of order 1.  Our
    daily-MAE stream has magnitude ~0.04, which is why an untreated stream never
    triggers: ADWIN misses even a 3x jump from 0.02 to 0.06 because the
    Hoeffding bound it compares against is far larger than the whole signal.

    Standardising fixes this and makes the thresholds scale-free, so the same
    delta/lambda mean the same thing for any target or any site.  The statistics
    are *frozen* after warm-up rather than kept running: a running normaliser
    would quietly absorb the very shift the detector is supposed to notice.
    """

    def __init__(self, warmup: int = 90):
        self.warmup = warmup
        self.reset()

    def reset(self) -> None:
        self._buf: list[float] = []
        self._mu: float | None = None
        self._sigma: float = 1.0

    def transform(self, value: float) -> float | None:
        """Return the z-score, or ``None`` while still warming up."""
        if self._mu is None:
            self._buf.append(value)
            if len(self._buf) < self.warmup:
                return None
            arr = np.asarray(self._buf, dtype=float)
            self._mu = float(arr.mean())
            self._sigma = float(max(arr.std(ddof=1), 1e-9))
            return 0.0
        return (value - self._mu) / self._sigma


class BaseDetector:
    """Detectors consume a *standardised* stream unless told otherwise."""

    name = "base"

    def __init__(self, standardize: bool = True, warmup: int = 90):
        self._std = ReferenceStandardizer(warmup) if standardize else None

    def _update_raw(self, value: float) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def _reset_state(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, value: float) -> bool:
        if self._std is not None:
            z = self._std.transform(value)
            if z is None:          # still warming up: no verdict yet
                return False
            value = z
        return self._update_raw(value)

    def reset(self) -> None:
        """Reset the detector but keep the frozen reference statistics.

        After a retrain the *detector* must start fresh, while the reference
        scale stays put -- otherwise each reset would re-baseline onto the
        drifted regime and the next shift would look normal.
        """
        self._reset_state()

    def hard_reset(self) -> None:
        self._reset_state()
        if self._std is not None:
            self._std.reset()

    def run(self, stream: pd.Series) -> list[DriftEvent]:
        self.hard_reset()
        events: list[DriftEvent] = []
        for i, (ts, v) in enumerate(stream.items()):
            if not np.isfinite(v):
                continue
            if self.update(float(v)):
                events.append(DriftEvent(i, ts, self.name, float(v)))
                self.reset()
        return events


class ADWINDetector(BaseDetector):
    """Adaptive windowing: split the recent stream and test the two means.

    Uses ``river``'s implementation when available; the fallback is a compact
    exponential-histogram-free version that compares an expanding reference
    window against a recent window with a Hoeffding-style bound, which is
    adequate for an hourly error stream of this length.
    """

    name = "adwin"

    def __init__(self, delta: float = 0.002, min_window: int = 96,
                 standardize: bool = True, warmup: int = 90):
        super().__init__(standardize=standardize, warmup=warmup)
        self.delta = delta
        self.min_window = min_window
        self._reset_state()

    def _reset_state(self) -> None:
        self._impl = _RiverADWIN(delta=self.delta) if _HAS_RIVER else None
        self._buf: list[float] = []

    def _update_raw(self, value: float) -> bool:
        if self._impl is not None:
            self._impl.update(value)
            return bool(self._impl.drift_detected)
        self._buf.append(value)
        if len(self._buf) < 2 * self.min_window:
            return False
        arr = np.asarray(self._buf)
        for cut in range(self.min_window, len(arr) - self.min_window + 1, self.min_window // 2):
            w0, w1 = arr[:cut], arr[cut:]
            m = 1.0 / (1.0 / len(w0) + 1.0 / len(w1))
            eps = np.sqrt(2.0 / m * np.log(4.0 / self.delta) * (arr.var() + 1e-12))
            if abs(w0.mean() - w1.mean()) > eps:
                return True
        return False


class PageHinkley(BaseDetector):
    """Cumulative deviation from the running mean; fires on a sustained shift.

    ``delta`` is the magnitude of change treated as noise and ``lambda_`` the
    alarm threshold.  Tuned deliberately conservatively: a detector that fires
    every fortnight would exhaust the retraining budget on weather noise.
    """

    name = "page_hinkley"

    def __init__(self, delta: float = 0.05, lambda_: float = 8.0, alpha: float = 0.9999,
                 min_instances: int = 60, standardize: bool = True, warmup: int = 90):
        super().__init__(standardize=standardize, warmup=warmup)
        self.delta, self.lambda_, self.alpha = delta, lambda_, alpha
        self.min_instances = min_instances
        self._reset_state()

    def _reset_state(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._sum = 0.0
        self._min_sum = 0.0

    def _update_raw(self, value: float) -> bool:
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum = self._sum * self.alpha + (value - self._mean - self.delta)
        self._min_sum = min(self._min_sum, self._sum)
        if self._n < self.min_instances:
            return False
        return (self._sum - self._min_sum) > self.lambda_


def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between two samples; >0.2 is the conventional "material shift" line.

    Unlike the error-stream detectors this looks at the *inputs*, so it can warn
    before performance degrades -- useful when labels arrive with a delay.
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins * 5 or len(current) < bins * 5:
        return 0.0
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(reference: np.ndarray, current: np.ndarray, alpha: float = 0.01
             ) -> tuple[bool, float, float]:
    """Two-sample KS test; returns (drift, statistic, p-value)."""
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 30 or len(current) < 30:
        return False, 0.0, 1.0
    stat, p = stats.ks_2samp(reference, current)
    return bool(p < alpha), float(stat), float(p)


@dataclass
class FeatureDriftReport:
    psi: pd.Series
    ks_stat: pd.Series
    ks_pvalue: pd.Series
    flagged: list[str] = field(default_factory=list)


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  columns: list[str] | None = None, psi_threshold: float = 0.2,
                  bins: int = 10) -> FeatureDriftReport:
    """Per-feature PSI and KS between a reference and a recent window."""
    columns = columns or [c for c in reference.columns
                          if pd.api.types.is_numeric_dtype(reference[c])]
    psi, ks_s, ks_p = {}, {}, {}
    for c in columns:
        a = reference[c].to_numpy(dtype=float)
        b = current[c].to_numpy(dtype=float)
        psi[c] = population_stability_index(a, b, bins)
        _, s, p = ks_drift(a, b)
        ks_s[c], ks_p[c] = s, p
    psi_s = pd.Series(psi).sort_values(ascending=False)
    return FeatureDriftReport(
        psi=psi_s, ks_stat=pd.Series(ks_s), ks_pvalue=pd.Series(ks_p),
        flagged=psi_s[psi_s > psi_threshold].index.tolist(),
    )


def daily_error_stream(predictions: pd.DataFrame, metric: str = "abs") -> pd.Series:
    """Collapse hourly forecasts to a daily error series for the detectors.

    Hourly errors are dominated by the diurnal cycle -- a detector run on them
    mostly rediscovers that night is easy.  Daily aggregation removes that and
    leaves the signal the detectors are meant to see.
    """
    err = predictions["y"] - predictions["point"]
    err = err.abs() if metric == "abs" else err
    if "daylight" in predictions.columns:
        err = err[predictions["daylight"].to_numpy()]
    return err.groupby(err.index.date).mean().rename("daily_error")
