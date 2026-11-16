"""Physics-informed features: solar position and clear-sky irradiance.

The case study deliberately runs *without* numerical weather prediction (NWP)
input, which is the realistic situation for a small operator who cannot buy a
commercial forecast feed.  The deterministic part of a PV series is however
almost entirely astronomy, and astronomy is free.  We therefore compute the
solar geometry analytically and let the learner spend its capacity on the
*stochastic* residual (the clear-sky index) instead of re-learning the sun.

References
----------
NOAA Solar Calculator equations (Michalsky 1988 / Spencer 1971 series).
Haurwitz, B. (1945). "Insolation in relation to cloudiness and cloud density."
Ineichen, P. & Perez, R. (2002). "A new airmass independent formulation for
    the Linke turbidity coefficient."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1367.0  # W/m^2


def _fractional_year(index: pd.DatetimeIndex) -> np.ndarray:
    doy = index.dayofyear.to_numpy(dtype=float)
    hour = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
    days_in_year = np.where(np.asarray(index.is_leap_year), 366.0, 365.0)
    return 2.0 * np.pi / days_in_year * (doy - 1.0 + (hour - 12.0) / 24.0)


def equation_of_time(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes of offset between apparent and mean solar time."""
    g = _fractional_year(index)
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )


def declination(index: pd.DatetimeIndex) -> np.ndarray:
    """Solar declination in radians."""
    g = _fractional_year(index)
    return (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.001480 * np.sin(3 * g)
    )


def solar_position(index: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Apparent solar zenith/elevation/azimuth for a UTC index.

    ``index`` must be timezone-aware UTC (or naive UTC).  Longitude is positive
    east of Greenwich.
    """
    if index.tz is not None:
        index = index.tz_convert("UTC")
    eot = equation_of_time(index)
    decl = declination(index)

    minutes = (
        index.hour.to_numpy(dtype=float) * 60.0
        + index.minute.to_numpy(dtype=float)
        + index.second.to_numpy(dtype=float) / 60.0
    )
    # Hour-angle is evaluated at the *centre* of the hourly averaging window,
    # which matters for the sunrise/sunset hours where irradiance is convex.
    true_solar_time = minutes + 30.0 + eot + 4.0 * longitude
    hour_angle = np.deg2rad((true_solar_time / 4.0) - 180.0)

    lat = np.deg2rad(latitude)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.arccos(cos_zenith)
    elevation = np.pi / 2.0 - zenith

    azimuth = np.arctan2(
        -np.sin(hour_angle),
        np.tan(decl) * np.cos(lat) - np.sin(lat) * np.cos(hour_angle),
    )
    azimuth = np.mod(azimuth, 2 * np.pi)

    return pd.DataFrame(
        {
            "solar_zenith": zenith,
            "solar_elevation": elevation,
            "solar_azimuth": azimuth,
            "cos_zenith": np.clip(cos_zenith, 0.0, 1.0),
            "declination": decl,
            "hour_angle": hour_angle,
        },
        index=index,
    )


def airmass(cos_zenith: np.ndarray) -> np.ndarray:
    """Kasten & Young (1989) relative airmass, guarded below the horizon."""
    cz = np.clip(cos_zenith, 1e-3, 1.0)
    zenith_deg = np.rad2deg(np.arccos(cz))
    return 1.0 / (cz + 0.50572 * (96.07995 - zenith_deg) ** -1.6364)


def clearsky_haurwitz(cos_zenith: np.ndarray) -> np.ndarray:
    """Haurwitz (1945) clear-sky global horizontal irradiance, W/m^2."""
    cz = np.clip(cos_zenith, 0.0, 1.0)
    ghi = 1098.0 * cz * np.exp(-0.059 / np.where(cz > 0, cz, 1.0))
    return np.where(cz > 0, ghi, 0.0)


def clearsky_ineichen(cos_zenith: np.ndarray, linke_turbidity: float = 3.0,
                      altitude_m: float = 200.0) -> np.ndarray:
    """Simplified Ineichen-Perez clear-sky GHI, W/m^2.

    A second clear-sky model is kept on purpose: the ratio of the two is a
    cheap proxy for how much of the model error is attributable to the
    clear-sky assumption itself rather than to cloud.
    """
    cz = np.clip(cos_zenith, 0.0, 1.0)
    am = airmass(cz)
    fh1 = np.exp(-altitude_m / 8000.0)
    fh2 = np.exp(-altitude_m / 1250.0)
    cg1 = 5.09e-05 * altitude_m + 0.868
    cg2 = 3.92e-05 * altitude_m + 0.0387
    ghi = (
        cg1
        * SOLAR_CONSTANT
        * cz
        * np.exp(-cg2 * am * (fh1 + fh2 * (linke_turbidity - 1.0)))
        * np.exp(0.01 * am ** 1.8)
    )
    return np.where(cz > 0, np.clip(ghi, 0.0, None), 0.0)


def clearsky_frame(index: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Full physical feature block for a timestamp index."""
    pos = solar_position(index, latitude, longitude)
    cz = pos["cos_zenith"].to_numpy()
    out = pos.copy()
    out["clearsky_haurwitz"] = clearsky_haurwitz(cz)
    out["clearsky_ineichen"] = clearsky_ineichen(cz)
    out["airmass"] = np.where(cz > 1e-3, airmass(cz), np.nan)
    out["is_daylight"] = (cz > 1e-3).astype(float)

    # Daily potential: integrates the geometry over the day, a strong seasonal
    # regressor that is exactly known 24 h in advance.
    daily = out["clearsky_haurwitz"].groupby(out.index.date).transform("sum")
    out["clearsky_daily_total"] = daily.to_numpy()
    return out


def clearsky_index(observed: pd.Series, clearsky: pd.Series,
                   floor: float = 1.0, clip: float = 1.5) -> pd.Series:
    """k_t = observed / clear-sky, defined only where the sun is meaningfully up.

    Normalising by the clear-sky expectation removes the deterministic diurnal
    and seasonal cycle, leaving a series whose conditional distribution is far
    closer to stationary -- which is exactly what a quantile learner needs.
    """
    denom = clearsky.where(clearsky > floor)
    k = (observed / denom).clip(upper=clip)
    return k
