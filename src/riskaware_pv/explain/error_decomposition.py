"""Where the error actually lives.

A single MAE hides everything an operator needs: whether the model is biased,
whether it fails at long lead times, whether it fails on cloudy days, and
whether its failures are concentrated in the hours that cost money.  Each
function here answers one of those questions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _err(preds: pd.DataFrame) -> pd.DataFrame:
    d = preds.copy()
    d["error"] = d["point"] - d["y"]
    d["abs_error"] = d["error"].abs()
    return d


def by_horizon(preds: pd.DataFrame, daylight_only: bool = True) -> pd.DataFrame:
    d = _err(preds)
    if daylight_only and "daylight" in d:
        d = d[d["daylight"]]
    g = d.groupby("horizon")
    return pd.DataFrame({"mae": g["abs_error"].mean(), "bias": g["error"].mean(),
                         "rmse": g["error"].apply(lambda e: np.sqrt((e ** 2).mean())),
                         "n": g.size()})


def by_hour_of_day(preds: pd.DataFrame) -> pd.DataFrame:
    d = _err(preds)
    d["hour"] = d.index.hour
    g = d.groupby("hour")
    return pd.DataFrame({"mae": g["abs_error"].mean(), "bias": g["error"].mean(),
                         "mean_y": g["y"].mean(), "n": g.size()})


def by_month(preds: pd.DataFrame, daylight_only: bool = True) -> pd.DataFrame:
    d = _err(preds)
    if daylight_only and "daylight" in d:
        d = d[d["daylight"]]
    d["month"] = d.index.month
    g = d.groupby("month")
    return pd.DataFrame({"mae": g["abs_error"].mean(), "bias": g["error"].mean(),
                         "mean_y": g["y"].mean(), "n": g.size()})


def by_regime(preds: pd.DataFrame, X: pd.DataFrame, n_bins: int = 4,
              column: str = "k_roll24_mean") -> pd.DataFrame:
    """Error against the cloudiness state known at the forecast origin.

    If error rises monotonically from clear to overcast regimes, the model is
    behaving sensibly and the remaining error is irreducible cloud noise; if it
    is flat, the model is ignoring the regime information it was given.
    """
    d = _err(preds)
    if "daylight" in d:
        d = d[d["daylight"]]
    key = X.reindex(d.index)[column]
    labels = [f"Q{i+1}" for i in range(n_bins)]
    d["regime"] = pd.qcut(key, n_bins, labels=labels, duplicates="drop")
    g = d.groupby("regime", observed=True)
    return pd.DataFrame({"mae": g["abs_error"].mean(), "bias": g["error"].mean(),
                         "mean_y": g["y"].mean(), "n": g.size()})


def bias_variance_split(preds: pd.DataFrame, by: str = "horizon") -> pd.DataFrame:
    """Decompose MSE into squared bias and error variance per group.

    Bias is fixable by recalibration; variance usually is not.  Knowing the
    split tells you whether to reach for a post-hoc correction or for better
    inputs.
    """
    d = _err(preds)
    if "daylight" in d:
        d = d[d["daylight"]]
    grouper = d.index.hour if by == "hour" else d[by]
    g = d.groupby(grouper)["error"]
    bias = g.mean()
    var = g.var(ddof=0)
    return pd.DataFrame({"bias": bias, "bias_sq": bias ** 2, "variance": var,
                         "mse": bias ** 2 + var}).rename_axis(by)


def worst_days(preds: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The days that dominate the aggregate error -- worth eyeballing by hand."""
    d = _err(preds)
    if "daylight" in d:
        d = d[d["daylight"]]
    g = d.groupby(d.index.normalize())
    out = pd.DataFrame({"mae": g["abs_error"].mean(), "bias": g["error"].mean(),
                        "mean_y": g["y"].mean(), "peak_y": g["y"].max()})
    return out.sort_values("mae", ascending=False).head(n)


def error_concentration(preds: pd.DataFrame) -> pd.Series:
    """What share of total absolute error comes from the worst k% of hours?

    PV forecast error is extremely concentrated.  Reporting this stops a reader
    from imagining the MAE is spread evenly across the year.
    """
    d = _err(preds)
    if "daylight" in d:
        d = d[d["daylight"]]
    e = np.sort(d["abs_error"].to_numpy())[::-1]
    total = e.sum()
    out = {}
    for pct in (1, 5, 10, 25, 50):
        k = max(1, int(len(e) * pct / 100))
        out[f"top_{pct}pct_share"] = float(e[:k].sum() / (total + 1e-12))
    return pd.Series(out)
