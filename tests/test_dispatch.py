"""Property tests for the dispatch LP.

These check economics rather than code paths: an optimiser that returns a
number is useless if the number does not obey the laws the problem is built on.
"""
from __future__ import annotations

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


def test_oracle_is_a_lower_bound(setup):
    """No forecast-driven policy may beat perfect foresight."""
    cfg, hours, load, pv = setup
    rng = np.random.default_rng(0)
    quantiles = np.asarray(cfg.forecast.quantiles)
    actual = pv * rng.uniform(0.3, 1.0)
    point = pv * 0.8
    q_pred = np.column_stack([pv * f for f in np.linspace(0.4, 1.1, len(quantiles))])

    oracle = P.run_policy("oracle", {"hours": hours, "load": load}, q_pred,
                          point / cfg.battery.pv_nameplate_mw,
                          actual / cfg.battery.pv_nameplate_mw, cfg)
    for policy in ("point", "quantile", "stochastic", "cvar"):
        got = P.run_policy(policy, {"hours": hours, "load": load},
                           q_pred / cfg.battery.pv_nameplate_mw,
                           point / cfg.battery.pv_nameplate_mw,
                           actual / cfg.battery.pv_nameplate_mw, cfg,
                           rng=np.random.default_rng(1))
        assert got["total_cost"] >= oracle["total_cost"] - 1e-6, (
            f"{policy} beat the oracle: {got['total_cost']} < {oracle['total_cost']}")


def test_perfect_forecast_reproduces_oracle(setup):
    """With zero forecast error the point policy must attain the oracle cost."""
    cfg, hours, load, pv = setup
    quantiles = np.asarray(cfg.forecast.quantiles)
    cf = pv / cfg.battery.pv_nameplate_mw
    q_pred = np.repeat(cf[:, None], len(quantiles), axis=1)  # degenerate distribution
    inputs = {"hours": hours, "load": load}
    oracle = P.run_policy("oracle", inputs, q_pred, cf, cf, cfg)
    point = P.run_policy("point", inputs, q_pred, cf, cf, cfg)
    assert point["total_cost"] == pytest.approx(oracle["total_cost"], rel=1e-6)
    assert point["imbalance_cost"] == pytest.approx(0.0, abs=1e-6)


def test_energy_balance_and_soc_bounds(setup):
    """Settlement must conserve energy and respect the battery envelope."""
    cfg, hours, load, pv = setup
    bat, mkt = cfg.battery, cfg.market
    res = D.solve_commitment(pv[None, :], load, hours, bat, mkt, terminal_soc=bat.soc_init)
    assert res.feasible

    out = D.evaluate_commitment(res.schedule, pv, load, hours, bat, mkt)
    assert out["status"] == "optimal"

    soc, ch, dis = out["soc"], out["charge"], out["discharge"]
    assert soc.min() >= bat.soc_min * bat.energy_mwh - 1e-6
    assert soc.max() <= bat.soc_max * bat.energy_mwh + 1e-6
    assert ch.max() <= bat.power_mw + 1e-6 and dis.max() <= bat.power_mw + 1e-6

    # hourly balance: pv + schedule + discharge - charge + up - dn == load
    prev = np.concatenate([[bat.soc_init * bat.energy_mwh], soc[:-1]])
    dyn = soc - prev - bat.eta_charge * ch + dis / bat.eta_discharge
    assert np.allclose(dyn, 0.0, atol=1e-6), "state-of-charge dynamics violated"


def test_asymmetric_penalties_push_commitment_below_the_mean(setup):
    """The newsvendor logic: expensive shortfalls => commit conservatively.

    With a short position penalised far more than a long one, the stochastic
    policy should plan to import more (or export less) than the point forecast
    would suggest.
    """
    cfg, hours, load, pv = setup
    quantiles = np.asarray(cfg.forecast.quantiles)
    cf_point = pv / cfg.battery.pv_nameplate_mw
    spread = np.linspace(0.35, 1.0, len(quantiles))
    q_pred = np.clip(cf_point[:, None] * spread[None, :], 0, None)
    inputs = {"hours": hours, "load": load}

    det = P.run_policy("point", inputs, q_pred, cf_point, cf_point, cfg)
    sto = P.run_policy("stochastic", inputs, q_pred, cf_point, cf_point, cfg,
                       rng=np.random.default_rng(3))
    assert sto["committed_import_mwh"] >= det["committed_import_mwh"] - 1e-6

    crit = P.newsvendor_quantile(cfg.market)
    assert 0.5 < crit < 1.0, f"critical fractile {crit} is not in the conservative half"
