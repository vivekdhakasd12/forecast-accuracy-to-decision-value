"""German market prices for the commitment decision: what is known at the gate
and what is paid at settlement.

Why this module exists
----------------------
German imbalance settlement (reBAP) uses one price for short and long.  If the
commitment problem simply assumes that price equals the day-ahead price, every
commitment has the same expected cost and the forecast stops mattering (code
review finding lp-1).

But the "site" in this thesis is the German solar fleet.  When German solar
falls short of what was forecast, the system is short and reBAP sits *above*
the day-ahead price; when it overshoots, reBAP sits *below*.  So for this
portfolio:

    short by x MWh  costs  (DA + pi_short) * x
    long  by x MWh  earns  (DA - pi_long)  * x

with ``pi_short`` and ``pi_long`` the average spread between reBAP and DA,
conditional on the direction of the solar surprise.  They are measured on
training years only and by German clock hour, because the spread has a clear
daily pattern.  That restores a newsvendor-type decision with a data-driven
cost ratio, which is the decision the thesis studies.

What is known when
------------------
* **Gate, 12:00 on day D** (commitment): the day-ahead prices for day D, which
  cleared on D-1.  Tomorrow's prices clear at about 12:42 on D, after the
  gate, so the commitment uses day D's price for the same hour as its estimate.
  The penalties come from the training period.
* **Settlement, after day D+1**: the realised day-ahead price and the realised
  reBAP.  These are only ever passed to settlement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dispatch import DayPrices

TZ = "Europe/Berlin"


@dataclass
class Penalties:
    """Per-German-hour penalties (EUR/MWh), indexed 0..23."""

    short: pd.Series
    long: pd.Series
    overall_short: float
    overall_long: float
    train_end: pd.Timestamp

    def at(self, local_hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.asarray(local_hours, dtype=int)
        return self.short.reindex(h).to_numpy(), self.long.reindex(h).to_numpy()

    def critical_fractile(self) -> pd.Series:
        """Cost-optimal commitment quantile per hour, ignoring the battery."""
        return self.long / (self.short + self.long)


def calibrate_penalties(panel: pd.DataFrame, train_end: str | pd.Timestamp,
                        min_mw: float = 1000.0, rel_band: float = 0.02,
                        shrink: float = 500.0, floor: float = 1.0) -> Penalties:
    """Measure the imbalance penalties from history up to ``train_end``.

    The direction of the system is read from the German solar surprise,
    ``solar_mw - solar_forecast_tso_mw``: an hour counts as *short* when actual
    solar fell more than ``rel_band`` below the TSO day-ahead forecast, and as
    *long* when it rose more than that above it.  Both columns are only used
    here, on the training window, never as model features.

    Hours with few solar observations (dawn, dusk) give noisy averages, so each
    hour is shrunk toward the overall average with weight ``shrink`` samples.
    Every penalty is floored at ``floor`` so being wrong is never free.
    """
    end = pd.Timestamp(train_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    p = panel.loc[:end, ["solar_mw", "solar_forecast_tso_mw", "price_eur_mwh",
                         "imbalance_price_eur_mwh"]].dropna()
    p = p[p["solar_forecast_tso_mw"] > min_mw]
    if p.empty:
        raise ValueError("no training hours with solar and prices before train_end")
    surprise = p["solar_mw"] - p["solar_forecast_tso_mw"]
    spread = p["imbalance_price_eur_mwh"] - p["price_eur_mwh"]
    band = rel_band * p["solar_forecast_tso_mw"]
    is_short, is_long = surprise < -band, surprise > band
    hour = p.index.tz_convert(TZ).hour

    overall_short = float(spread[is_short].mean())
    overall_long = float(-spread[is_long].mean())
    hours = pd.Index(range(24), name="hour")

    def by_hour(mask: pd.Series, sign: float, overall: float) -> pd.Series:
        s = (sign * spread[mask]).groupby(hour[mask.to_numpy()])
        mean, n = s.mean().reindex(hours), s.size().reindex(hours).fillna(0)
        shrunk = (n * mean.fillna(0.0) + shrink * overall) / (n + shrink)
        return shrunk.clip(lower=floor)

    return Penalties(short=by_hour(is_short, 1.0, overall_short),
                     long=by_hour(is_long, -1.0, overall_long),
                     overall_short=overall_short, overall_long=overall_long,
                     train_end=end)


def _local_hours(index: pd.DatetimeIndex) -> np.ndarray:
    return index.tz_convert(TZ).hour.to_numpy()


def gate_day_ahead_estimate(panel: pd.DataFrame, delivery: pd.DatetimeIndex) -> np.ndarray:
    """Day-ahead price estimate available at the gate for each delivery hour.

    Uses the price of the same hour one day earlier (day D), which cleared on
    D-1 and is therefore known at the 12:00 gate on D.  On the 25-hour autumn
    day, stepping back 24 hours can land inside the delivery day itself, which
    is not yet known; those hours step back 48 hours instead.
    """
    delivery = pd.DatetimeIndex(delivery)
    local_start = delivery.tz_convert(TZ).tz_localize(None).normalize().min()
    day_start_utc = local_start.tz_localize(TZ).tz_convert("UTC")
    back = delivery - pd.Timedelta(hours=24)
    back = back.where(back < day_start_utc, delivery - pd.Timedelta(hours=48))
    est = panel["price_eur_mwh"].reindex(back).to_numpy()
    if np.isnan(est).any():
        raise ValueError("missing gate-time day-ahead price for some delivery hours")
    return est


def gate_prices(panel: pd.DataFrame, delivery: pd.DatetimeIndex,
                penalties: Penalties) -> DayPrices:
    """Prices the commitment is decided on: nothing here is unknown at the gate."""
    da_est = gate_day_ahead_estimate(panel, delivery)
    ps, pl = penalties.at(_local_hours(pd.DatetimeIndex(delivery)))
    return DayPrices(import_price=da_est, export_price=da_est.copy(),
                     short_price=da_est + ps, long_price=-(da_est - pl))


def settlement_prices(panel: pd.DataFrame, delivery: pd.DatetimeIndex) -> DayPrices:
    """Realised prices: day-ahead revenue and single-price reBAP imbalance."""
    rows = panel.reindex(pd.DatetimeIndex(delivery))
    da = rows["price_eur_mwh"].to_numpy()
    imb = rows["imbalance_price_eur_mwh"].to_numpy()
    if np.isnan(da).any() or np.isnan(imb).any():
        raise ValueError("delivery day has missing realised prices; skip it explicitly")
    return DayPrices.single_price(da, imb)
