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


def _local_day_start(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    """Local calendar day of each timestamp, as tz-naive local midnight."""
    return index.tz_convert(timezone).tz_localize(None).normalize()


def forecast_origin(index: pd.DatetimeIndex, gate_hour: int,
                    timezone: str = "UTC") -> pd.DatetimeIndex:
    """Gate timestamp (in UTC) that governs each target timestamp.

    Delivery day D+1 is decided at ``gate_hour`` on day D, both read on the
    clock of ``timezone``.  With ``"UTC"`` this is exactly the Case Study 1
    protocol.  With ``"Europe/Berlin"`` the gate is 12:00 German time, which is
    11:00 UTC in winter and 10:00 UTC in summer, and day D+1 is a German
    calendar day.
    """
    if timezone == "UTC" or index.tz is None:
        day_start = index.normalize()
        return day_start - pd.Timedelta(days=1) + pd.Timedelta(hours=gate_hour)
    # Do the calendar arithmetic on the local wall clock, then localise: adding
    # "one day" to an aware timestamp adds 24 hours, which is wrong on the two
    # clock-change days.  12:00 is never skipped or repeated by a clock change.
    gate_local = (_local_day_start(index, timezone) - pd.Timedelta(days=1)
                  + pd.Timedelta(hours=gate_hour))
    return gate_local.tz_localize(timezone).tz_convert("UTC")


def delivery_day_hours(origin: pd.DatetimeIndex, timezone: str = "UTC") -> np.ndarray:
    """Number of hours in the delivery day that each origin decides (23, 24 or 25)."""
    if timezone == "UTC" or origin.tz is None:
        return np.full(len(origin), 24)
    day = _local_day_start(origin, timezone) + pd.Timedelta(days=1)
    start = day.tz_localize(timezone)
    end = (day + pd.Timedelta(days=1)).tz_localize(timezone)
    return ((end - start) / pd.Timedelta(hours=1)).astype(int).to_numpy()


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

    ``origin`` here is the *availability cutoff*: the latest timestamp whose
    value is complete at the gate (gate minus the availability lag).
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

    tz = getattr(fc, "timezone", "UTC")
    origin = forecast_origin(idx, fc.gate_hour, tz)
    horizon = ((idx - origin) / pd.Timedelta(hours=1)).astype(float)
    # Latest timestamp whose value is complete at the gate.  Every historical
    # feature below reads at or before this cutoff, never at the gate itself.
    cutoff = origin - pd.Timedelta(hours=getattr(fc, "availability_lag_hours", 0))

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
        X[f"y_sameh_d{d}"] = _gated_shift(y, idx, cutoff, d)
        X[f"k_sameh_d{d}"] = _gated_shift(k_index, idx, cutoff, d)
    X["y_sameh_mean_d123"] = X[[f"y_sameh_d{d}" for d in (1, 2, 3)]].mean(axis=1)
    X["y_sameh_std_d123"] = X[[f"y_sameh_d{d}" for d in (1, 2, 3)]].std(axis=1)
    # Neighbouring hours of the most recent legal day capture cloud structure.
    for off in (-2, -1, 1, 2):
        X[f"y_sameh_d1_off{off:+d}"] = _gated_shift(y, idx, cutoff, 1, hour_offset=off)

    # ---- state-at-origin block: "how cloudy has it been lately" ------------
    for w in cfg.forecast.rolling_windows:
        X[f"k_roll{w}_mean"] = _rolling_at_origin(k_index, cutoff, w, "mean")
        X[f"k_roll{w}_std"] = _rolling_at_origin(k_index, cutoff, w, "std")
        X[f"y_roll{w}_mean"] = _rolling_at_origin(y, cutoff, w, "mean")
    X["k_recent6"] = _rolling_at_origin(k_index, cutoff, 6, "mean")
    X["y_at_origin"] = y.reindex(cutoff).to_numpy()

    # ---- exogenous block (also gated) --------------------------------------
    for exo, alias in (("load_mw", "load"), ("price_eur_mwh", "price"),
                       ("wind_mw", "wind")):
        if exo in frame.columns:
            s = frame[exo].astype(float)
            X[f"{alias}_sameh_d1"] = _gated_shift(s, idx, cutoff, 1)
            X[f"{alias}_roll168_mean"] = _rolling_at_origin(s, cutoff, 168, "mean")

    X["y"] = y.to_numpy()
    X["k"] = k_index.to_numpy()
    X["capacity_mw"] = frame.get("capacity_mw", pd.Series(1.0, index=idx)).to_numpy()

    # Drop rows whose gated history does not exist yet (start of sample) and
    # rows outside the legal day-ahead horizon band.
    # Delivery day D+1 hour u sits (24 - gate_hour + u) hours after the gate,
    # so the legal band is [24 - gate, 47 - gate] for a 24-hour delivery day.
    # (On local time the band is 23 to 25 hours wide around clock changes, and
    # every row already belongs to the day after its own gate by construction.)
    if tz == "UTC":
        lo, hi = 24 - fc.gate_hour, 24 - fc.gate_hour + fc.horizon - 1
        X = X[(X["horizon"] >= lo) & (X["horizon"] <= hi)]
    X = X.dropna(subset=["y", "y_sameh_d1", "y_sameh_d7", "k_roll168_mean"])

    # Keep only complete delivery days.  The downstream dispatch problem is
    # posed over the whole delivery day, so a day missing three hours is not a
    # slightly worse sample -- it is a different optimisation problem.  A local
    # delivery day has 23, 24 or 25 hours; a UTC one always has 24.
    size = X.groupby("origin")["y"].transform("size").to_numpy()
    if tz == "UTC":
        expected = fc.horizon
    else:
        expected = delivery_day_hours(pd.DatetimeIndex(X["origin"]), tz)
    return X[size == expected]


def feature_columns(X: pd.DataFrame) -> list[str]:
    """Model inputs: everything except targets, bookkeeping and raw scale."""
    drop = {"y", "k", "origin", "capacity_mw"}
    return [c for c in X.columns if c not in drop]


def night_mask(X: pd.DataFrame, threshold: float = 1e-3) -> np.ndarray:
    """Hours with no meaningful sun -- predicted as exactly zero, not learned."""
    return (X["cos_zenith"].to_numpy() <= threshold)
