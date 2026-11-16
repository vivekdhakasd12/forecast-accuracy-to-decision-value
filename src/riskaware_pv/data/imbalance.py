"""German imbalance price (reBAP) from Netztransparenz, the four German TSOs.

reBAP ("regelzonenübergreifender einheitlicher Bilanzausgleichsenergiepreis") is
the price a balance responsible party pays or receives for every MWh it is off
its schedule, per quarter hour.  Germany uses a single price: the same number
applies whether you are short or long.  In practice that means

    cost of being short by x MWh  =  +reBAP * x
    cost of being long  by x MWh  =  -reBAP * x     (you are paid reBAP)

The published file has two columns, "reBAP unterdeckt" (short) and
"reBAP ueberdeckt" (long).  They are identical except in the rare quarter hours
where the capacity reserve was activated, so both are kept.

Two ways to get the data:

1. **API (automatic).**  Register for free at https://www.netztransparenz.de
   (Web-API section), create a client, and set two environment variables::

       export NTP_CLIENT_ID=...
       export NTP_CLIENT_SECRET=...

   The loader then fetches and caches month by month under ``data/raw/rebap``.

2. **Manual CSV.**  Download reBAP CSVs from the Netztransparenz website and
   drop them in ``data/raw/rebap/``.  Any ``*.csv`` there is read.

Endpoint, as documented in the Netztransparenz Web-API documentation v1.21:
``GET https://ds.netztransparenz.de/api/v1/data/NrvSaldo/reBAP/Qualitaetsgesichert/{from}/{to}``
with an OAuth 2.0 client-credentials token from
``https://identity.netztransparenz.de/users/connect/token``.

The imbalance price is a *settlement* quantity.  It is only known well after
delivery, so it must never be used as a model feature; ``features.py`` does not
read it.
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_RAW, ROOT

TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
API_BASE = "https://ds.netztransparenz.de/api/v1/data/NrvSaldo/reBAP/Qualitaetsgesichert"
RAW_DIR = DATA_RAW / "rebap"
ENV_FILE = ROOT / ".env"


def _load_env_file(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines from the private ``.env`` file into the environment.

    The file holds the Netztransparenz API keys.  It is listed in .gitignore,
    so it stays on this computer.  Variables already set in the shell win.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_env_file()


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _num(s: pd.Series) -> pd.Series:
    """German number format: '1.234,56' -> 1234.56."""
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False),
        errors="coerce")


def parse_rebap_csv(text: str) -> pd.DataFrame:
    """Parse a Netztransparenz reBAP CSV (format 9) into a quarter-hourly frame.

    Expected header::

        Datum;Zeitzone;von;bis;Datenkategorie;Datentyp;Einheit;reBAP unterdeckt;reBAP ueberdeckt

    Returns a frame indexed by the UTC start of each quarter hour with columns
    ``rebap_short_eur_mwh`` and ``rebap_long_eur_mwh``.
    """
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    short_col = next(c for c in df.columns if "unterdeckt" in c.lower())
    long_col = next(c for c in df.columns if "berdeckt" in c.lower() and c != short_col)
    tz = df["Zeitzone"].str.strip().str.upper()
    if not (tz == "UTC").all():
        raise ValueError(f"unexpected time zones in reBAP file: {sorted(tz.unique())}")
    # Netztransparenz writes dd.mm.yyyy; some exports use ISO yyyy-mm-dd.  Parse
    # each explicitly: "mixed" + dayfirst would read 2023-06-10 as 6 October.
    raw_day = df["Datum"].str.strip()
    iso = raw_day.str.match(r"^\d{4}-\d{2}-\d{2}$")
    day = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    day[iso] = pd.to_datetime(raw_day[iso], format="%Y-%m-%d")
    day[~iso] = pd.to_datetime(raw_day[~iso], format="%d.%m.%Y")
    hhmm = df["von"].str.strip().str.split(":", expand=True).astype(int)
    start = day + pd.to_timedelta(hhmm[0], unit="h") + pd.to_timedelta(hhmm[1], unit="m")
    out = pd.DataFrame({
        "rebap_short_eur_mwh": _num(df[short_col]).to_numpy(),
        "rebap_long_eur_mwh": _num(df[long_col]).to_numpy(),
    }, index=pd.DatetimeIndex(start).tz_localize("UTC"))
    out.index.name = "utc_timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def _token(client_id: str, client_secret: str) -> str:
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(TOKEN_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())["access_token"]


def download_rebap(start: str, end: str, refresh: bool = False) -> list[Path]:
    """Fetch reBAP month by month via the API and cache each month as CSV."""
    cid, secret = os.environ.get("NTP_CLIENT_ID"), os.environ.get("NTP_CLIENT_SECRET")
    if not cid or not secret:
        raise FileNotFoundError(
            "No reBAP data: set NTP_CLIENT_ID and NTP_CLIENT_SECRET (free Netztransparenz "
            "Web-API registration) or put downloaded reBAP CSVs in " + str(RAW_DIR))
    token = _token(cid, secret)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    months = pd.date_range(pd.Timestamp(start).to_period("M").start_time,
                           pd.Timestamp(end), freq="MS")
    for m in months:
        path = RAW_DIR / f"rebap_{m:%Y-%m}.csv"
        paths.append(path)
        if path.exists() and not refresh:
            continue
        nxt = m + pd.offsets.MonthBegin(1)
        url = f"{API_BASE}/{m:%Y-%m-%dT%H:%M:%S}/{nxt:%Y-%m-%dT%H:%M:%S}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    return paths


def load_rebap(start: str, end: str) -> pd.DataFrame:
    """Quarter-hourly reBAP over [start, end], from cached CSVs or the API."""
    if os.environ.get("NTP_CLIENT_ID") and os.environ.get("NTP_CLIENT_SECRET"):
        download_rebap(start, end)          # cached months are skipped
    files = sorted(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
    if not files:
        download_rebap(start, end)          # raises a clear FileNotFoundError
    frames = []
    for f in files:
        raw = f.read_bytes()
        for enc in ("utf-8-sig", "latin-1"):
            try:
                frames.append(parse_rebap_csv(raw.decode(enc)))
                break
            except UnicodeDecodeError:
                continue
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")
                   + pd.Timedelta(minutes=45)]


def load_rebap_hourly(start: str, end: str, column: str = "rebap_short_eur_mwh") -> pd.Series:
    """Hourly mean of the quarter-hourly price.

    This is exact when the deviation is constant within the hour, which is the
    case for an hourly schedule settled against hourly generation.  Settling at
    quarter-hour resolution is a later refinement and should use
    :func:`load_rebap` directly.
    """
    qh = load_rebap(start, end)[column]
    hourly = qh.resample("h").mean()
    counts = qh.resample("h").count()
    hourly[counts < 4] = np.nan          # an incomplete hour is not a price
    hourly.name = "imbalance_price_eur_mwh"
    return hourly
