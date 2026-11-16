"""Gate-time and settlement prices for the German market (decision/market.py).

The key property: under these prices a better forecast changes the promise.
Under the earlier single-price "gate view" every promise cost the same, so the
solver picked one at random (code review finding lp-1).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from riskaware_pv.config import Config
from riskaware_pv.decision import dispatch as D
from riskaware_pv.decision import market as M
from riskaware_pv.decision import policies as P


def _panel(days: int = 400, start: str = "2021-01-01", seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly panel where reBAP sits above DA when solar falls short."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    hour = idx.tz_convert("Europe/Berlin").hour.to_numpy()
    sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 40_000
    forecast = sun * rng.uniform(0.7, 1.1, len(idx))
    actual = forecast * rng.normal(1.0, 0.15, len(idx)).clip(0.3, 1.7)
    da = 80 + 30 * np.cos(hour / 24 * 2 * np.pi) + rng.normal(0, 10, len(idx))
    surprise = (actual - forecast) / 40_000
    imb = da - 400 * surprise + rng.normal(0, 5, len(idx))   # short -> higher price
    return pd.DataFrame({"solar_mw": actual, "solar_forecast_tso_mw": forecast,
                         "price_eur_mwh": da, "imbalance_price_eur_mwh": imb}, index=idx)


def _delivery_day(day: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day).tz_localize("Europe/Berlin")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Berlin")
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _generator_cfg() -> Config:
    """A plain solar generator: no own load, no battery."""
    cfg = copy.deepcopy(Config())
    cfg.battery.energy_mwh = 1e-6
    cfg.battery.power_mw = 0.0
    cfg.battery.soc_min, cfg.battery.soc_max, cfg.battery.soc_init = 0.0, 1.0, 0.0
    return cfg


def test_penalties_are_positive_and_only_use_training_data():
    panel = _panel()
    pen = M.calibrate_penalties(panel, "2021-09-30")
    assert (pen.short >= 1.0).all() and (pen.long >= 1.0).all()
    assert pen.overall_short > 0 and pen.overall_long > 0
    poisoned = panel.copy()
    poisoned.loc["2021-10-01":, "imbalance_price_eur_mwh"] = 1e6
    pen2 = M.calibrate_penalties(poisoned, "2021-09-30")
    pd.testing.assert_series_equal(pen.short, pen2.short)
    pd.testing.assert_series_equal(pen.long, pen2.long)


def test_gate_estimate_never_reads_the_delivery_day():
    panel = _panel()
    day = _delivery_day("2021-06-10")
    est = M.gate_day_ahead_estimate(panel, day)
    assert np.allclose(est, panel["price_eur_mwh"].reindex(day - pd.Timedelta(hours=24)))
    poisoned = panel.copy()
    poisoned.loc[day, "price_eur_mwh"] = -9999.0
    assert np.allclose(est, M.gate_day_ahead_estimate(poisoned, day))


def test_gate_estimate_on_the_25_hour_day():
    panel = _panel()
    day = _delivery_day("2021-10-31")
    assert len(day) == 25
    poisoned = panel.copy()
    poisoned.loc[day, "price_eur_mwh"] = -9999.0
    est = M.gate_day_ahead_estimate(poisoned, day)
    assert (est > -9999.0).all()                      # nothing read from the day itself


def test_gate_prices_pass_the_safety_check():
    panel = _panel()
    pen = M.calibrate_penalties(panel, "2021-09-30")
    gp = M.gate_prices(panel, _delivery_day("2021-11-15"), pen)
    assert gp.speculation_hours().size == 0


def test_the_forecast_now_changes_the_promise():
    """Regression for lp-1: different forecasts must give different schedules."""
    panel = _panel()
    cfg = _generator_cfg()
    pen = M.calibrate_penalties(panel, "2021-09-30")
    day = _delivery_day("2021-11-15")
    gp = M.gate_prices(panel, day, pen)
    hours = np.arange(len(day))
    load = np.zeros(len(day))
    pv = 2.0 * np.clip(np.sin((hours - 7) / 10 * np.pi), 0, None)
    schedules = []
    for f in (0.5, 1.0):
        res = D.solve_commitment((f * pv)[None, :], load, hours, cfg.battery, cfg.market,
                                 prices=gp, terminal_soc=0.0)
        assert res.feasible
        schedules.append(res.schedule)
        # with one certain scenario the best promise is exactly the forecast
        assert np.allclose(res.schedule, -f * pv, atol=1e-6)
    assert not np.allclose(schedules[0], schedules[1])


def test_stochastic_commitment_follows_the_measured_cost_ratio():
    """Under uncertainty the promise sits at the hour's critical fractile."""
    cfg = _generator_cfg()
    hours = np.arange(24)
    load = np.zeros(24)
    scen = np.linspace(0.0, 2.0, 201)[:, None] * np.ones((1, 24))   # uniform 0..2 MW
    ps, pl = 30.0, 10.0                                              # short is costlier
    da = np.full(24, 60.0)
    gp = D.DayPrices.dual_price(da, ps, pl)
    res = D.solve_commitment(scen, load, hours, cfg.battery, cfg.market, prices=gp,
                             terminal_soc=0.0)
    tau = pl / (ps + pl)                                             # = 0.25
    assert np.allclose(-res.schedule, 2.0 * tau, atol=0.02)


def test_settlement_reports_honest_short_and_long_energy():
    """Regression for lp-2: no phantom short+long in the same hour."""
    panel = _panel()
    cfg = Config()
    pen = M.calibrate_penalties(panel, "2021-09-30")
    day = _delivery_day("2021-11-15")
    hours = np.arange(len(day))
    load = P.default_load_profile(hours, cfg.battery)
    pv = 2.0 * np.clip(np.sin((hours - 7) / 10 * np.pi), 0, None)
    gp, sp = M.gate_prices(panel, day, pen), M.settlement_prices(panel, day)
    res = D.solve_commitment((0.9 * pv)[None, :], load, hours, cfg.battery, cfg.market,
                             prices=gp, terminal_soc=0.5)
    out = D.evaluate_commitment(res.schedule, 0.6 * pv, load, hours, cfg.battery,
                                cfg.market, terminal_soc=0.5, prices=gp, settle_prices=sp)
    gross = out["energy_short_mwh"] + out["energy_long_mwh"]
    # the battery can only shrink the mismatch, never invent extra imbalance
    net = np.abs(load - 0.6 * pv - res.schedule).sum() + cfg.battery.energy_mwh
    assert gross <= net + 1e-6


@pytest.mark.parametrize("prices", [
    D.DayPrices(np.full(3, 300.0), np.full(3, 50.0), np.full(3, 200.0), np.full(3, 10.0)),
    D.DayPrices(np.full(3, 300.0), np.full(3, 50.0), np.full(3, 400.0), np.full(3, -100.0)),
])
def test_safety_check_catches_the_two_missing_cases(prices):
    """Regression for lp-3: short cheaper than buying, long dearer than selling."""
    assert prices.speculation_hours().size == 3


def test_missing_realised_prices_are_not_silently_settled():
    panel = _panel()
    day = _delivery_day("2021-11-15")
    panel.loc[day[5], "imbalance_price_eur_mwh"] = np.nan
    with pytest.raises(ValueError):
        M.settlement_prices(panel, day)
