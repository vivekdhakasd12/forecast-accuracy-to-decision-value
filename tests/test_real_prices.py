"""The dispatch LP under real market prices.

Case Study 1 only ever saw one positive, fixed tariff.  Real German prices go
negative (the SMARD series hits -500 EUR/MWh) and German imbalance settlement
uses a single price, so a long position gets *paid*.  Under those prices the
original LP had unbounded variables and simply had no solution.  These tests
pin down that it now always solves, that it still respects the economics, and
that nothing changed for the Case Study 1 tariff.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from riskaware_pv.config import Config
from riskaware_pv.decision import dispatch as D
from riskaware_pv.decision import policies as P


@pytest.fixture
def setup():
    cfg = Config()
    hours = np.arange(24)
    load = P.default_load_profile(hours, cfg.battery)
    pv = cfg.battery.pv_nameplate_mw * np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)
    return cfg, hours, load, pv


def _real_like_prices(hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A sunny-Sunday shape: deeply negative at midday, high in the evening."""
    da = 80.0 - 140.0 * np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) \
        + 60.0 * (hours >= 18)
    imb = da + np.where(hours % 2 == 0, 35.0, -50.0)
    return da, imb


def test_cs1_tariff_unchanged(setup):
    """Default prices == explicit CS1 prices, and the new bounds never bind.

    Only the planned *cost* is compared: the LP has tied optima, so the
    schedule itself may legitimately differ once bounds are added.
    """
    cfg, hours, load, pv = setup
    scen = np.vstack([pv * f for f in (0.4, 0.7, 1.0)])
    a = D.solve_commitment(scen, load, hours, cfg.battery, cfg.market, terminal_soc=0.5)
    b = D.solve_commitment(scen, load, hours, cfg.battery, cfg.market, terminal_soc=0.5,
                           prices=D.DayPrices.from_market(hours, cfg.market))
    loose = copy.deepcopy(cfg.battery)
    loose.grid_import_max_mw = loose.grid_export_max_mw = 1e6
    c = D.solve_commitment(scen, load, hours, loose, cfg.market, terminal_soc=0.5)
    assert a.feasible and b.feasible and c.feasible
    assert a.objective == pytest.approx(b.objective, rel=1e-9)
    assert a.objective == pytest.approx(c.objective, rel=1e-7)


def test_old_unbounded_case_now_solves(setup):
    """Export paid more than the short penalty: CS1's LP was unbounded here."""
    cfg, hours, load, pv = setup
    mkt = copy.deepcopy(cfg.market)
    mkt.export_price = 350.0
    # speculation has to be allowed explicitly; the point here is boundedness,
    # which holds whenever prices are passed explicitly
    res = D.solve_commitment(pv[None, :], load, hours, cfg.battery, mkt, terminal_soc=0.5,
                             prices=D.DayPrices.from_market(hours, mkt),
                             allow_speculation=True)
    assert res.feasible and np.isfinite(res.objective)


def test_speculative_commitment_prices_are_refused(setup):
    """Committing with the realised imbalance price lets the LP bet on the
    DA/imbalance spread instead of forecasting.  That must be refused."""
    cfg, hours, load, pv = setup
    da, imb = _real_like_prices(hours)
    with pytest.raises(ValueError, match="deliberate imbalance"):
        D.solve_commitment(pv[None, :], load, hours, cfg.battery, cfg.market,
                           prices=D.DayPrices.single_price(da, imb))
    assert D.DayPrices.gate_view(da).speculation_hours().size == 0
    assert D.DayPrices.from_market(hours, cfg.market).speculation_hours().size == 0
    assert D.DayPrices.dual_price(da, 50.0, 20.0).speculation_hours().size == 0


@pytest.mark.parametrize("policy", ["oracle", "point", "quantile", "stochastic", "cvar"])
def test_every_policy_solves_with_negative_real_prices(setup, policy):
    cfg, hours, load, pv = setup
    da, imb = _real_like_prices(hours)
    assert da.min() < 0 and imb.min() < 0
    nameplate = cfg.battery.pv_nameplate_mw
    q = np.asarray(cfg.forecast.quantiles)
    q_pred = np.column_stack([pv * f for f in np.linspace(0.4, 1.1, len(q))]) / nameplate
    out = P.run_policy(policy, {"hours": hours, "load": load}, q_pred,
                       0.8 * pv / nameplate, 0.6 * pv / nameplate, cfg,
                       rng=np.random.default_rng(3),
                       prices=D.DayPrices.gate_view(da),
                       settle_prices=D.DayPrices.single_price(da, imb))
    assert out["status"] == "optimal"
    assert np.isfinite(out["total_cost"])


def test_oracle_still_a_lower_bound_under_real_prices(setup):
    cfg, hours, load, pv = setup
    prices = D.DayPrices.dual_price(_real_like_prices(hours)[0], 80.0, 30.0)
    nameplate = cfg.battery.pv_nameplate_mw
    q = np.asarray(cfg.forecast.quantiles)
    q_pred = np.column_stack([pv * f for f in np.linspace(0.4, 1.1, len(q))]) / nameplate
    args = ({"hours": hours, "load": load}, q_pred, 0.9 * pv / nameplate,
            0.5 * pv / nameplate, cfg)
    oracle = P.run_policy("oracle", *args, prices=prices)["total_cost"]
    for pol in ("point", "quantile", "stochastic"):
        got = P.run_policy(pol, *args, rng=np.random.default_rng(1), prices=prices)
        assert got["total_cost"] >= oracle - 1e-6, pol


def test_single_price_long_is_paid(setup):
    """Under single pricing, delivering more than committed earns the imbalance price."""
    cfg, hours, load, pv = setup
    bat = copy.deepcopy(cfg.battery)
    bat.energy_mwh = 1e-6                     # no storage: isolate settlement
    bat.soc_min, bat.soc_max, bat.power_mw = 0.0, 1.0, 0.0
    imb = np.full(24, 100.0)
    prices = D.DayPrices.single_price(np.full(24, 50.0), imb)
    schedule = np.zeros(24)                   # commit nothing, deliver PV + cover load
    out = D.evaluate_commitment(schedule, pv, load, hours, bat, cfg.market, prices=prices)
    net = pv - load                           # positive = long, negative = short
    expected = float(np.sum(-net * imb))
    assert out["imbalance_cost"] == pytest.approx(expected, rel=1e-6)


def test_nan_prices_are_rejected(setup):
    _, hours, _, _ = setup
    da = np.full(24, 50.0)
    da[5] = np.nan
    with pytest.raises(ValueError):
        D.DayPrices.single_price(da, np.full(24, 60.0))


def test_wrong_length_prices_are_rejected(setup):
    cfg, hours, load, pv = setup
    prices = D.DayPrices.single_price(np.full(23, 50.0), np.full(23, 60.0))
    with pytest.raises(ValueError):
        D.solve_commitment(pv[None, :], load, hours, cfg.battery, cfg.market, prices=prices)


def test_failed_solve_raises_instead_of_nan(setup, monkeypatch):
    """A failed day must stop the run, not vanish from the average."""
    cfg, hours, load, pv = setup
    failed = D.DispatchResult(status="unbounded", objective=np.inf, schedule=np.zeros(24),
                              import_mwh=np.zeros(24), export_mwh=np.zeros(24))
    monkeypatch.setattr(D, "solve_commitment", lambda *a, **k: failed)
    cf = pv / cfg.battery.pv_nameplate_mw
    q_pred = np.repeat(cf[:, None], len(cfg.forecast.quantiles), axis=1)
    with pytest.raises(RuntimeError):
        P.run_policy("point", {"hours": hours, "load": load}, q_pred, cf, cf, cfg)
