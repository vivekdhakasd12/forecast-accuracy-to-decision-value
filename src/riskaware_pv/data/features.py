"""Leakage-safe day-ahead feature construction.

The single most common way to accidentally fake skill in a day-ahead study is
to use a lag that would not exist at gate closure.  With a 12:00 gate, the
"obvious" ``y[t-24]`` feature is *illegal* for every target hour after 12:00 of
the delivery day, because that observation is only made after we had to submit.

This module therefore never uses a fixed lag.  For each target timestamp ``t``
it derives the forecast origin, then selects the most recent same-hour-of-day
observation that is genuinely available at that origin:

    k_min = ceil(horizon / 24)      ->   y[t - 24 * k_min]  is always legal

Every historical feature is built through :func:`_gated_shift`, and
``tests/test_no_leakage.py`` verifies the property empirically by poisoning the
series after each origin and asserting the feature matrix is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from . import solar_geometry as sg

FOURIER_ORDERS = (1, 2, 3)


def forecast_origin(index: pd.DatetimeIndex, gate_hour: int) -> pd.DatetimeIndex:
    """Gate timestamp that governs each target timestamp.

    Delivery day D+1 (all 24 hours) is decided at ``gate_hour`` on day D.
    """
    day_start = index.normalize()
    return day_start - pd.Timedelta(days=1) + pd.Timedelta(hours=gate_hour)


def _gated_shift(series: pd.Series, target_index: pd.DatetimeIndex,
                 origin: pd.DatetimeIndex, days_back: int,
                 hour_offset: int = 0) -> np.ndarray:
    """Same-hour-of-day value from ``days_back`` *available* days ago.

    ``days_back=1`` means "the most recent legal same-hour observation",
    ``days_back=2`` the one before it, and so on.  The number of calendar days
    skipped adapts to the horizon so the result can never post-date ``origin``.

    ``hour_offset`` shifts the requested hour within that day (used for the
    neighbouring-hour cloud-structure features).  It is folded into the day
    count *before* rounding, because a naive "+1 hour" on a lag that already
    sits exactly at the gate would step one hour past it -- a leak small enough
    to survive code review and large enough to inflate reported skill.
    """
    horizon_h = (target_index - origin) / pd.Timedelta(hours=1)
    k_min = np.ceil((horizon_h + hour_offset) / 24.0)
    offsets = (k_min + (days_back - 1)) * 24.0 - hour_offset
    wanted = target_index - pd.to_timedelta(offsets, unit="h")
    assert (wanted <= origin).all(), "gated shift produced a future-looking lag"
    return series.reindex(wanted).to_numpy()


def _rolling_at_origin(series: pd.Series, origin: pd.DatetimeIndex,
                       window_hours: int, stat: str = "mean") -> np.ndarray:
    """Statistic of the last ``window_hours`` ending *at* the origin (inclusive)."""
    roll = getattr(series.rolling(window_hours, min_periods=max(2, window_hours // 4)), stat)()
    return roll.reindex(origin).to_numpy()


def _fourier(values: np.ndarray, period: float, orders=FOURIER_ORDERS) -> dict[str, np.ndarray]:
    out = {}
    for k in orders:
        ang = 2.0 * np.pi * k * values / period
        out[f"sin{k}"] = np.sin(ang)
        out[f"cos{k}"] = np.cos(ang)
    return out


def build_design_matrix(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build the (target x features) table for the day-ahead problem.

    Parameters
    ----------
    frame
        Output of :func:`riskaware_pv.data.opsd.load_site_series`; must contain
        a ``target`` column normalised to a capacity factor and a UTC index.
    """
    fc, site = cfg.forecast, cfg.site
    idx = pd.DatetimeIndex(frame.index)
    y = frame["target"].astype(float)

    phys = sg.clearsky_frame(idx, site.latitude, site.longitude)
    cs_norm = phys["clearsky_haurwitz"] / max(phys["clearsky_haurwitz"].max(), 1e-9)
    k_index = sg.clearsky_index(y, cs_norm, floor=0.02)

    origin = forecast_origin(idx, fc.gate_hour)
    horizon = ((idx - origin) / pd.Timedelta(hours=1)).astype(float)

    X = pd.DataFrame(index=idx)
    X["origin"] = origin
    X["horizon"] = horizon

    # ---- deterministic block: known exactly, arbitrarily far ahead ----------
    for col in ("cos_zenith", "solar_elevation", "solar_azimuth", "airmass",
                "clearsky_haurwitz", "clearsky_ineichen", "clearsky_daily_total",
                "is_daylight"):
        X[col] = phys[col].to_numpy()
    X["clearsky_norm"] = cs_norm.to_numpy()
    # Guard the denominator at 1 W/m2 rather than at zero.  Around sunrise and
    # sunset the Haurwitz value passes through arbitrarily small positives, and
    # dividing by those produced ratios up to 4e197 -- finite, so no NaN check
    # caught them, and harmless to LightGBM, but XGBoost's histogram builder
    # rejects the matrix outright.  Below 1 W/m2 the ratio carries no
    # information anyway, so it is left missing.
    denom = phys["clearsky_haurwitz"].where(phys["clearsky_haurwitz"] > 1.0)
    X["clearsky_ratio"] = (phys["clearsky_ineichen"] / denom).to_numpy()

    hours = idx.hour.to_numpy(dtype=float)
    doy = idx.dayofyear.to_numpy(dtype=float)
    for name, vals in _fourier(hours, 24.0).items():
        X[f"hod_{name}"] = vals
    for name, vals in _fourier(doy, 365.25, orders=(1, 2)).items():
        X[f"doy_{name}"] = vals
    X["hour"] = hours
    X["dayofweek"] = idx.dayofweek.to_numpy(dtype=float)
    X["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    # ---- autoregressive block: gated so nothing post-dates the origin ------
    for d in (1, 2, 3, 7):
        X[f"y_sameh_d{d}"] = _gated_shift(y, idx, origin, d)
        X[f"k_sameh_d{d}"] = _gated_shift(k_index, idx, origin, d)
    X["y_sameh_mean_d123"] = X[[f"y_sameh_d{d}" for d in (1, 2, 3)]].mean(axis=1)
    X["y_sameh_std_d123"] = X[[f"y_sameh_d{d}" for d in (1, 2, 3)]].std(axis=1)
    # Neighbouring hours of the most recent legal day capture cloud structure.
    for off in (-2, -1, 1, 2):
        X[f"y_sameh_d1_off{off:+d}"] = _gated_shift(y, idx, origin, 1, hour_offset=off)

    # ---- state-at-origin block: "how cloudy has it been lately" ------------
    for w in cfg.forecast.rolling_windows:
        X[f"k_roll{w}_mean"] = _rolling_at_origin(k_index, origin, w, "mean")
        X[f"k_roll{w}_std"] = _rolling_at_origin(k_index, origin, w, "std")
        X[f"y_roll{w}_mean"] = _rolling_at_origin(y, origin, w, "mean")
    X["k_recent6"] = _rolling_at_origin(k_index, origin, 6, "mean")
    X["y_at_origin"] = y.reindex(origin).to_numpy()

    # ---- exogenous block (also gated) --------------------------------------
    for exo, alias in (("load_mw", "load"), ("price_eur_mwh", "price"),
                       ("wind_mw", "wind")):
        if exo in frame.columns:
            s = frame[exo].astype(float)
            X[f"{alias}_sameh_d1"] = _gated_shift(s, idx, origin, 1)
            X[f"{alias}_roll168_mean"] = _rolling_at_origin(s, origin, 168, "mean")

    X["y"] = y.to_numpy()
    X["k"] = k_index.to_numpy()
    X["capacity_mw"] = frame.get("capacity_mw", pd.Series(1.0, index=idx)).to_numpy()

    # Drop rows whose gated history does not exist yet (start of sample) and
    # rows outside the legal day-ahead horizon band.
    # Delivery day D+1 hour u sits (24 - gate_hour + u) hours after the gate,
    # so the legal band is [24 - gate, 47 - gate] for a 24-hour delivery day.
    lo, hi = 24 - fc.gate_hour, 24 - fc.gate_hour + fc.horizon - 1
    X = X[(X["horizon"] >= lo) & (X["horizon"] <= hi)]
    X = X.dropna(subset=["y", "y_sameh_d1", "y_sameh_d7", "k_roll168_mean"])

    # Keep only complete delivery days.  The downstream dispatch problem is
    # posed over a full 24-hour commitment window, so a day missing three hours
    # is not a slightly worse sample -- it is a different optimisation problem.
    complete = X.groupby("origin")["y"].transform("size") == fc.horizon
    return X[complete]


def feature_columns(X: pd.DataFrame) -> list[str]:
    """Model inputs: everything except targets, bookkeeping and raw scale."""
    drop = {"y", "k", "origin", "capacity_mw"}
    return [c for c in X.columns if c not in drop]


def night_mask(X: pd.DataFrame, threshold: float = 1e-3) -> np.ndarray:
    """Hours with no meaningful sun -- predicted as exactly zero, not learned."""
    return (X["cos_zenith"].to_numpy() <= threshold)
