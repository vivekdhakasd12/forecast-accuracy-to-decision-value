"""German generation, load and day-ahead price from SMARD (Bundesnetzagentur).

Why this exists: the OPSD package that Case Study 1 used stops on 2020-09-30,
and the German day-ahead price in it only starts at the 2018 bidding-zone split.
The thesis needs real prices over a longer, more recent window, so this module
rebuilds the German panel from SMARD, which is free, needs no account, and is
published under CC BY 4.0.

    https://www.smard.de  (Bundesnetzagentur | SMARD.de)

Series used (SMARD "filter" ids):

======  ======  ===============================================  =========================
filter  region  meaning                                          column
======  ======  ===============================================  =========================
4068    DE      PV generation, realised                          ``solar_mw``
125     DE      PV generation, TSO day-ahead forecast            ``solar_forecast_tso_mw``
410     DE      total load (Netzlast)                            ``load_mw``
4169    DE-LU   day-ahead price, DE-LU bidding zone              ``price_eur_mwh``
======  ======  ===============================================  =========================

Installed PV capacity (for the capacity-factor target) comes from the
Energy-Charts API of Fraunhofer ISE, monthly, in GW (DC nameplate), which is the
same convention as the OPSD capacity column used in Case Study 1.

Two columns are *not* allowed to become features, and ``features.py`` never
reads them: ``solar_forecast_tso_mw`` (published after the 12:00 gate, so it
is only fair as a benchmark), and the imbalance price (settled long after
delivery).  They ride along in the panel so that evaluation can use them.

Every downloaded file is cached under ``data/raw/smard`` and
``data/raw/energy_charts``, so a rebuild never hits the network twice.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_INTERIM, DATA_RAW
from .opsd import _longest_valid_span, _quality_control

SMARD_BASE = "https://www.smard.de/app/chart_data"
ENERGY_CHARTS_CAPACITY = (
    "https://api.energy-charts.info/installed_power"
    "?country=de&time_step=monthly&installation_decommission=false"
)

SMARD_SERIES: dict[str, tuple[int, str]] = {
    "solar_mw": (4068, "DE"),
    "solar_forecast_tso_mw": (125, "DE"),
    "load_mw": (410, "DE"),
    "price_eur_mwh": (4169, "DE-LU"),
}

# Default study window.  It starts after the 2018 DE-LU split (one price
# definition throughout) and runs to the end of October 2026.
#
# Note on 2025-10-01: from that day the day-ahead market trades 15-minute
# products.  SMARD's hourly price after that date is the average of the four
# quarter-hour prices.  The thesis keeps an hourly schedule for the whole window
# and states this as an approximation; ``AUCTION_15MIN_FROM`` marks the switch
# so results can be reported separately for the two periods.
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2026-10-31 23:00"
AUCTION_15MIN_FROM = "2025-10-01"

RAW_DIR = DATA_RAW / "smard"
CAPACITY_JSON = DATA_RAW / "energy_charts" / "installed_power_de_monthly.json"
PANEL_PARQUET = DATA_INTERIM / "smard_de_panel.parquet"

_WEEK_MS = 7 * 24 * 3600 * 1000


# --------------------------------------------------------------------------- #
# low-level fetching
# --------------------------------------------------------------------------- #
def _get_json(url: str, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "riskaware-pv"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network hiccup: back off and retry
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def _cached_json(url: str, path: Path, refresh: bool = False) -> dict:
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = _get_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def parse_smard_series(payload: dict) -> pd.Series:
    """Turn one SMARD chart-data file into a UTC-indexed float series.

    SMARD stores ``[epoch_ms, value]`` pairs, where the timestamp is the
    *start* of the interval and missing values are ``null``.
    """
    pairs = payload.get("series") or []
    if not pairs:
        return pd.Series(dtype=float)
    ts, vals = zip(*pairs)
    idx = pd.to_datetime(np.asarray(ts, dtype="int64"), unit="ms", utc=True)
    return pd.Series(np.asarray([np.nan if v is None else v for v in vals], dtype=float),
                     index=idx)


def fetch_smard_series(filter_id: int, region: str, start: str, end: str,
                       resolution: str = "hour", refresh: bool = False) -> pd.Series:
    """Download (or reuse) every weekly SMARD file that overlaps [start, end]."""
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC")
    index = _get_json(f"{SMARD_BASE}/{filter_id}/{region}/index_{resolution}.json")
    stamps = [s for s in index["timestamps"]
              if s + _WEEK_MS > t0.value // 10**6 and s <= t1.value // 10**6]
    parts = []
    for s in stamps:
        name = f"{filter_id}_{region}_{resolution}_{s}.json"
        url = f"{SMARD_BASE}/{filter_id}/{region}/{name}"
        # A week that is not over yet will still change: never cache it.
        complete = s + _WEEK_MS < pd.Timestamp.now(tz="UTC").value // 10**6 - 2 * 86_400_000
        payload = (_cached_json(url, RAW_DIR / name, refresh=refresh) if complete
                   else _get_json(url))
        parts.append(parse_smard_series(payload))
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[t0:t1]


def parse_capacity(payload: dict, name: str = "Solar DC") -> pd.Series:
    """Monthly installed capacity in MW, indexed by the first hour of each month (UTC)."""
    months = pd.to_datetime(payload["time"], format="%m.%Y")
    for item in payload["production_types"]:
        if item["name"] == name:
            gw = np.asarray([np.nan if v is None else v for v in item["data"]], dtype=float)
            return pd.Series(gw * 1000.0, index=months.tz_localize("UTC"), name="capacity_mw")
    raise KeyError(f"{name!r} not in Energy-Charts payload")


def fetch_solar_capacity(refresh: bool = False) -> pd.Series:
    return parse_capacity(_cached_json(ENERGY_CHARTS_CAPACITY, CAPACITY_JSON, refresh=refresh))


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
def build_de_panel(start: str = DEFAULT_START, end: str = DEFAULT_END,
                   force: bool = False, refresh: bool = False,
                   with_imbalance: bool = True) -> pd.DataFrame:
    """Hourly German panel on a gap-free UTC clock, cached as parquet."""
    if PANEL_PARQUET.exists() and not force and not refresh:
        cached = pd.read_parquet(PANEL_PARQUET)
        covers = (cached.index.min() <= pd.Timestamp(start, tz="UTC")
                  and cached.index.max() >= pd.Timestamp(end, tz="UTC"))
        # A panel built before the reBAP files existed has an empty imbalance
        # column; rebuild it once they are there instead of serving it forever.
        from .imbalance import RAW_DIR as REBAP_DIR
        stale = (with_imbalance and REBAP_DIR.exists() and any(REBAP_DIR.glob("*.csv"))
                 and cached.get("imbalance_price_eur_mwh", pd.Series(dtype=float)).isna().all())
        if covers and not stale:
            return cached.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]

    cols = {}
    for col, (fid, region) in SMARD_SERIES.items():
        cols[col] = fetch_smard_series(fid, region, start, end, refresh=refresh)
    full = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                         freq="h", name="utc_timestamp")
    df = pd.DataFrame({c: s.reindex(full) for c, s in cols.items()}, index=full)

    cap = fetch_solar_capacity(refresh=refresh)
    df["solar_capacity_mw"] = cap.reindex(full, method="ffill")

    if with_imbalance:
        from .imbalance import load_rebap_hourly  # local import: optional source
        try:
            df["imbalance_price_eur_mwh"] = load_rebap_hourly(start, end).reindex(full)
        except FileNotFoundError:
            df["imbalance_price_eur_mwh"] = np.nan

    # SMARD's price column carries a few hours of zeros/NaN at DST switches;
    # micro-gap interpolation in the shared QC handles them.
    df = _quality_control(df)
    PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_PARQUET)
    return df


def load_site_series_smard(target: str = "solar", start: str = DEFAULT_START,
                           end: str = DEFAULT_END, force: bool = False) -> pd.DataFrame:
    """Same contract as :func:`opsd.load_site_series`, built from SMARD.

    The target is a capacity factor: realised PV divided by installed DC
    capacity.  German PV more than doubled over 2019-2025, so the raw MW series
    would carry a large bookkeeping trend.
    """
    if target != "solar":
        raise KeyError("the SMARD panel is built for the solar target only")
    panel = build_de_panel(start, end, force=force)
    out = pd.DataFrame(index=panel.index)
    out["target_mw"] = panel["solar_mw"]
    out["capacity_mw"] = panel["solar_capacity_mw"]
    out["target"] = (panel["solar_mw"] / panel["solar_capacity_mw"]).clip(0.0, 1.2)
    for extra in ("load_mw", "price_eur_mwh", "solar_mw", "solar_forecast_tso_mw",
                  "imbalance_price_eur_mwh"):
        if extra in panel:
            out[extra] = panel[extra]
    out.attrs["normalisation"] = "capacity"
    out.attrs["country"] = "DE"
    out.attrs["source"] = "smard"
    s, e = _longest_valid_span(out["target"])
    return out.loc[s:e]


if __name__ == "__main__":  # pragma: no cover - manual build
    import argparse

    ap = argparse.ArgumentParser(description="Build the SMARD German panel")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    p = build_de_panel(a.start, a.end, force=a.force)
    print(p.describe().T.to_string())
    print(p.index.min(), "->", p.index.max(), len(p), "hours")
    print("missing share per column:\n", p.isna().mean().round(4).to_string())
