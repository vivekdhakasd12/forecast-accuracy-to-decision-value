"""Curtailment: the PV plant may switch off part of its output.

With real German prices going negative (SMARD has 510 such hours between
Oct 2025 and Oct 2026 alone), a plant that cannot switch off is forced to pay
to get rid of its energy.  Real plants in direct marketing switch off instead.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from riskaware_pv.config import ROOT, Config
from riskaware_pv.decision import dispatch as D


def _generator(curtail: bool) -> Config:
    cfg = copy.deepcopy(Config())
    cfg.battery.energy_mwh = 1e-6
    cfg.battery.power_mw = 0.0
    cfg.battery.soc_min, cfg.battery.soc_max, cfg.battery.soc_init = 0.0, 1.0, 0.0
    cfg.battery.allow_curtailment = curtail
    return cfg


HOURS = np.arange(24)
LOAD = np.zeros(24)
PV = 2.0 * np.clip(np.sin((HOURS - 6) / 12 * np.pi), 0, None)
NEG = (HOURS >= 10) & (HOURS <= 14)                 # a sunny Sunday: negative midday
DA = np.where(NEG, -80.0, 60.0)


def _prices() -> D.DayPrices:
    return D.DayPrices.dual_price(DA, penalty_short=30.0, penalty_long=20.0)


def test_thesis_config_switches_it_on_and_case_study_1_keeps_it_off():
    assert Config.load(ROOT / "configs" / "thesis.yaml").battery.allow_curtailment is True
    assert Config.load().battery.allow_curtailment is False


def test_plant_switches_off_instead_of_paying_to_export():
    cfg = _generator(curtail=True)
    res = D.solve_commitment(PV[None, :], LOAD, HOURS, cfg.battery, cfg.market,
                             prices=_prices(), terminal_soc=0.0)
    assert res.feasible
    # negative hours: sell nothing, throw the energy away
    assert np.allclose(res.schedule[NEG], 0.0, atol=1e-6)
    assert np.allclose(res.curtailed[0, NEG], PV[NEG], atol=1e-6)
    # positive hours: sell everything, curtail nothing
    assert np.allclose(res.curtailed[0, ~NEG], 0.0, atol=1e-6)
    assert np.allclose(res.schedule[~NEG], -PV[~NEG], atol=1e-6)


def test_switching_off_is_never_more_expensive():
    on, off = _generator(True), _generator(False)
    a = D.solve_commitment(PV[None, :], LOAD, HOURS, on.battery, on.market,
                           prices=_prices(), terminal_soc=0.0)
    b = D.solve_commitment(PV[None, :], LOAD, HOURS, off.battery, off.market,
                           prices=_prices(), terminal_soc=0.0)
    assert a.objective < b.objective - 1.0          # saves real money on this day


def test_settlement_curtails_an_unplanned_surplus_at_negative_prices():
    """Promised less than produced in a negative hour: switch off the surplus
    rather than be paid a negative price for it."""
    cfg = _generator(curtail=True)
    schedule = -0.5 * PV                             # promised half the output
    out = D.evaluate_commitment(schedule, PV, LOAD, HOURS, cfg.battery, cfg.market,
                                prices=_prices(), terminal_soc=0.0)
    assert out["status"] == "optimal"
    assert out["curtailed_mwh"] == pytest.approx(float((0.5 * PV[NEG]).sum()), rel=1e-6)


def test_no_curtailment_when_prices_are_positive():
    cfg = _generator(curtail=True)
    prices = D.DayPrices.dual_price(np.full(24, 60.0), 30.0, 20.0)
    res = D.solve_commitment(PV[None, :], LOAD, HOURS, cfg.battery, cfg.market,
                             prices=prices, terminal_soc=0.0)
    assert np.allclose(res.curtailed, 0.0, atol=1e-6)


def test_reported_cost_excludes_the_tiebreak():
    cfg = _generator(curtail=True)
    res = D.solve_commitment(PV[None, :], LOAD, HOURS, cfg.battery, cfg.market,
                             prices=_prices(), terminal_soc=0.0)
    p = _prices()
    expected = float(np.sum(np.maximum(res.schedule, 0) * p.import_price
                            - np.maximum(-res.schedule, 0) * p.export_price))
    assert res.objective == pytest.approx(expected, abs=1e-6)
