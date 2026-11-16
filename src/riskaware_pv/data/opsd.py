"""Ingest the Open Power System Data (OPSD) hourly time-series package.

Source: https://data.open-power-system-data.org/time_series/  (CC-BY 4.0)
The national German series are used because they are long (2015-2020), openly
licensed, hourly, and contain generation, load and day-ahead price on one
clock -- everything the downstream dispatch experiment needs.

The raw package is ~130 MB; we slice the handful of German columns we need and
persist a compact parquet so that every later stage is fast and deterministic.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_INTERIM, DATA_RAW

OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_60min_singleindex.csv"
)
RAW_CSV = DATA_RAW / "opsd_60min.csv"
PANEL_PARQUET = DATA_INTERIM / "opsd_de_panel.parquet"

# canonical name -> candidate OPSD column suffixes (first match wins).  The
# country prefix is substituted in, so the same map serves every zone.
COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "solar_mw": ("{c}_solar_generation_actual",),
    "wind_mw": ("{c}_wind_generation_actual", "{c}_wind_onshore_generation_actual"),
    "load_mw": ("{c}_load_actual_entsoe_transparency",
                "{c}_load_actual_entsoe_power_statistics"),
    "price_eur_mwh": ("{c}_price_day_ahead", "{c}_LU_price_day_ahead"),
    "solar_capacity_mw": ("{c}_solar_capacity",),
    "wind_capacity_mw": ("{c}_wind_capacity", "{c}_wind_onshore_capacity"),
}

# Zones used by the transfer study, with the approximate capacity-weighted
# centroid that the clear-sky model needs.  These are not political centroids:
# British PV sits overwhelmingly in the south of England, so using the
# geographic centre of Great Britain would put the sun in the wrong place by
# roughly two degrees of latitude.
SITES: dict[str, dict] = {
    "DE":     {"name": "Germany",       "latitude": 51.0, "longitude": 10.0},
    "CH":     {"name": "Switzerland",   "latitude": 46.9, "longitude": 8.2},
    "DK":     {"name": "Denmark",       "latitude": 56.0, "longitude": 10.0},
    "GB_GBN": {"name": "Great Britain", "latitude": 51.5, "longitude": -1.0},
}


def download(force: bool = False, url: str = OPSD_URL, dest: Path = RAW_CSV) -> Path:
    """Fetch the raw CSV once; subsequent calls are no-ops."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000_000 and not force:
        return dest
    with urllib.request.urlopen(url, timeout=900) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def _resolve_columns(header: list[str], country: str = "DE") -> dict[str, str]:
    resolved = {}
    for canonical, candidates in COLUMN_MAP.items():
        for cand in candidates:
            col = cand.format(c=country)
            if col in header:
                resolved[canonical] = col
                break
    return resolved


def build_panel(force: bool = False, country: str = "DE") -> pd.DataFrame:
    """Return the tidy German panel, building and caching it if necessary."""
    cache = (PANEL_PARQUET if country == "DE"
             else PANEL_PARQUET.with_name(f"opsd_{country}_panel.parquet"))
    if cache.exists() and not force:
        cached = pd.read_parquet(cache)
        return cached

    csv_path = download()
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    resolved = _resolve_columns(header, country)
    if "solar_mw" not in resolved:
        raise RuntimeError(
            f"OPSD file at {csv_path} lacks a solar column for {country}; "
            f"got {header[:12]}..."
        )

    usecols = ["utc_timestamp", *resolved.values()]
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["utc_timestamp"])
    df = df.rename(columns={v: k for k, v in resolved.items()})
    df = df.set_index("utc_timestamp").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "utc_timestamp"

    # Enforce a gap-free hourly clock so that lag arithmetic is positional-safe.
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full)
    df.index.name = "utc_timestamp"

    for cap_col in ("solar_capacity_mw", "wind_capacity_mw"):
        if cap_col in df:
            _CAPACITY_REPORTED[(country, cap_col)] = df[cap_col].notna()
    df = _quality_control(df)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# Peak generation above this multiple of reported capacity means the two
# series describe different scopes and capacity cannot be a denominator.
CAPACITY_COHERENCE_MAX = 1.15

_CAPACITY_REPORTED: dict[tuple[str, str], pd.Series] = {}


def _quality_control(df: pd.DataFrame) -> pd.DataFrame:
    """Clip physically impossible values and interpolate only micro-gaps.

    Long gaps are left as NaN on purpose: silently filling a week of missing
    generation would manufacture skill that the model does not have.
    """
    out = df.copy()
    for col in ("solar_mw", "wind_mw", "load_mw"):
        if col in out:
            out[col] = out[col].clip(lower=0.0)
    # capacity series are reported irregularly -> forward fill is correct here
    for col in ("solar_capacity_mw", "wind_capacity_mw"):
        if col in out:
            out[col] = out[col].ffill().bfill()
    for col in ("solar_mw", "wind_mw", "load_mw", "price_eur_mwh"):
        if col in out:
            out[col] = out[col].interpolate(limit=3, limit_area="inside")
    return out


def _longest_valid_span(series: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Longest contiguous run of non-NaN values (the usable modelling window)."""
    valid = series.notna().to_numpy()
    best_len = best_start = cur_start = 0
    for i, ok in enumerate(valid):
        if not ok:
            cur_start = i + 1
            continue
        if i - cur_start + 1 > best_len:
            best_len = i - cur_start + 1
            best_start = cur_start
    idx = series.index
    return idx[best_start], idx[best_start + best_len - 1]


def load_site_series(
    target: str = "solar",
    capacity_normalise: bool = True,
    force: bool = False,
    country: str = "DE",
    normalise: str = "capacity",
) -> pd.DataFrame:
    """Return the modelling frame: target plus the exogenous columns.

    ``capacity_normalise`` converts MW into a *capacity factor* in [0, 1] using
    OPSD's reported installed capacity.  This matters: German PV capacity roughly
    doubled over the sample, so an un-normalised target contains a large
    deterministic trend that would flatter any model with a trend term and would
    contaminate the concept-drift experiment with a purely bookkeeping shift.
    """
    panel = build_panel(force=force, country=country)

    col = {"solar": "solar_mw", "wind": "wind_mw", "load": "load_mw"}[target]
    if col not in panel:
        raise KeyError(f"target '{target}' unavailable; columns={list(panel.columns)}")

    out = pd.DataFrame(index=panel.index)
    out["target_mw"] = panel[col]

    cap_col = {"solar": "solar_capacity_mw", "wind": "wind_capacity_mw"}.get(target)
    have_cap = bool(cap_col and cap_col in panel and panel[cap_col].notna().any())
    coherent = False
    if have_cap:
        cap_ff = panel[cap_col].replace(0.0, np.nan).ffill().bfill()
        ratio = float((panel[col] / cap_ff).max())
        coherent = ratio <= CAPACITY_COHERENCE_MAX

    mode = normalise
    if mode == "auto":
        mode = "capacity" if coherent else "quantile"
    if mode == "capacity" and not have_cap:
        mode = "quantile"

    if mode == "capacity":
        cap = panel[cap_col].replace(0.0, np.nan)
        out["capacity_mw"] = cap
        out["target"] = (out["target_mw"] / cap).clip(0.0, 1.2)
        reported = _CAPACITY_REPORTED.get((country, cap_col))
        # Quality control forward-fills capacity, which is right *inside* the
        # reporting window and wrong outside it: several zones stop reporting
        # capacity while generation continues, and a frozen denominator under a
        # growing fleet inflates the capacity factor.
        if reported is not None and reported.any():
            idx = reported[reported].index
            out = out.loc[idx.min(): idx.max()]
    else:
        # Robust per-site scale.  Only Germany's reported capacity is coherent
        # with its own generation series in this package: Switzerland, Denmark
        # and Great Britain all report peak generation above reported capacity
        # (by 2.94x, 1.81x and 1.37x), so the capacity column describes a
        # different scope there and cannot serve as a denominator.  The 99.9th
        # percentile of generation is a well-defined alternative that puts every
        # site on a comparable [0, 1] scale, which is what the transfer study
        # needs; it is not a capacity factor and is not labelled as one.
        scale = float(panel[col].quantile(0.999))
        out["capacity_mw"] = scale
        out["target"] = (out["target_mw"] / max(scale, 1e-9)).clip(0.0, 1.2)

    out.attrs["normalisation"] = mode
    out.attrs["country"] = country

    for extra in ("load_mw", "price_eur_mwh", "wind_mw", "solar_mw"):
        if extra in panel:
            out[extra] = panel[extra]

    start, end = _longest_valid_span(out["target"])
    out = out.loc[start:end]
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    frame = load_site_series()
    print(frame.describe())
    print(frame.index.min(), "->", frame.index.max(), len(frame), "hours")
