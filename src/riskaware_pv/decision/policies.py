"""From a predictive distribution to a committed schedule.

Five commitment policies, deliberately spanning the range from "what most sites
actually do" to "what the full predictive distribution allows":

``oracle``       perfect foresight -- the lower bound that defines regret
``point``        commit to the mean/median forecast (the industry default)
``quantile``     commit to a fixed conservative quantile (a cheap hedge)
``stochastic``   scenarios drawn from the predictive quantiles, minimise expected cost
``cvar``         same scenarios, minimise a blend of expected cost and tail cost

The scenario construction is the interesting part.  Sampling each hour
independently from its marginal quantile function would produce physically
absurd days -- bright, dark, bright -- and would badly understate the cost of a
uniformly overcast day, which is exactly the event that hurts.  We therefore
sample a *single* uniform per day and read every hour's quantile function at
that level, plus a smaller independent perturbation.  This is a Gaussian-copula-
style dependence structure with one dominant common factor, which is a good
first-order description of daily cloud cover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BatteryConfig, Config, MarketConfig
from . import dispatch as D


def scenarios_from_quantiles(
    q_pred: np.ndarray,
    quantiles: np.ndarray,
    n_scenarios: int = 40,
    common_factor: float = 0.75,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw ``n_scenarios`` daily PV paths from hourly marginal quantiles.

    Parameters
    ----------
    q_pred : (24, n_quantiles)
        Hourly predictive quantiles for one delivery day.
    common_factor : float in [0, 1]
        Weight on the shared daily uniform.  1.0 gives perfectly rank-correlated
        hours (all-clear or all-cloudy days only); 0.0 gives independent hours.
    """
    rng = rng or np.random.default_rng(0)
    T = q_pred.shape[0]
    u_common = rng.uniform(size=(n_scenarios, 1))
    u_indep = rng.uniform(size=(n_scenarios, T))
    u = np.clip(common_factor * u_common + (1.0 - common_factor) * u_indep, 1e-3, 1 - 1e-3)
    out = np.empty((n_scenarios, T))
    for t in range(T):
        out[:, t] = np.interp(u[:, t], quantiles, np.sort(q_pred[t]))
    return np.clip(out, 0.0, None)


def _as_site_mw(capacity_factor: np.ndarray, battery: BatteryConfig) -> np.ndarray:
    """Rescale a national capacity factor to this site's nameplate."""
    return np.asarray(capacity_factor, dtype=float) * battery.pv_nameplate_mw


def build_day_inputs(day: pd.DataFrame, cfg: Config,
                     load_profile: np.ndarray | None = None) -> dict:
    """Assemble the per-day inputs shared by every policy."""
    hours = day.index.hour.to_numpy()
    order = np.argsort(hours)
    day = day.iloc[order]
    hours = hours[order]
    if load_profile is None:
        load = default_load_profile(hours, cfg.battery)
    else:
        load = np.asarray(load_profile, dtype=float)[order]
    return {"hours": hours, "load": load, "day": day}


def default_load_profile(hours: np.ndarray, battery: BatteryConfig) -> np.ndarray:
    """A commercial-site daily load shape (morning ramp, midday plateau, evening peak).

    Load is treated as known: this study isolates *PV* uncertainty, so mixing in
    an unmodelled load error would confound the attribution of dispatch regret.
    """
    base = 0.35 + 0.45 * np.exp(-0.5 * ((hours - 12.0) / 4.5) ** 2) \
        + 0.30 * np.exp(-0.5 * ((hours - 19.0) / 1.8) ** 2)
    return base / base.max() * battery.load_peak_mw


def run_policy(
    policy: str,
    day_inputs: dict,
    q_pred: np.ndarray,
    point: np.ndarray,
    actual: np.ndarray,
    cfg: Config,
    rng: np.random.Generator | None = None,
    fixed_quantile: float = 0.25,
) -> dict:
    """Commit under ``policy``, settle against ``actual``, return costs + regret."""
    quantiles = np.asarray(cfg.forecast.quantiles, dtype=float)
    bat, mkt = cfg.battery, cfg.market
    hours, load = day_inputs["hours"], day_inputs["load"]
    terminal = bat.soc_init

    pv_actual = _as_site_mw(actual, bat)

    if policy == "oracle":
        commit = D.solve_commitment(pv_actual[None, :], load, hours, bat, mkt,
                                    terminal_soc=terminal)
    elif policy == "point":
        commit = D.solve_commitment(_as_site_mw(point, bat)[None, :], load, hours,
                                    bat, mkt, terminal_soc=terminal)
    elif policy == "quantile":
        # Interpolate the predictive quantile function rather than snapping to the
        # nearest fitted level: snapping would make a sweep over commitment
        # quantiles report a spurious plateau wherever several requested levels
        # round to the same fitted column.
        qcurve = np.array([
            np.interp(fixed_quantile, quantiles, np.sort(q_pred[t]))
            for t in range(q_pred.shape[0])
        ])
        commit = D.solve_commitment(_as_site_mw(qcurve, bat)[None, :], load, hours,
                                    bat, mkt, terminal_soc=terminal)
    elif policy in ("stochastic", "cvar"):
        scen = scenarios_from_quantiles(q_pred, quantiles, mkt.n_scenarios, rng=rng)
        commit = D.solve_commitment(
            _as_site_mw(scen, bat), load, hours, bat, mkt,
            cvar_weight=0.5 if policy == "cvar" else 0.0,
            cvar_alpha=mkt.cvar_alpha, terminal_soc=terminal,
        )
    else:
        raise ValueError(f"unknown policy {policy!r}")

    if not commit.feasible:
        return {"policy": policy, "status": commit.status, "total_cost": np.nan}

    # The terminal state-of-charge obligation must be scored, not just planned.
    # Every commitment problem above enforces it; if settlement did not, a policy
    # that budgets for the obligation would pay for it in the day-ahead schedule
    # while a policy that ignored it would be settled for free -- an asymmetry
    # that grows with battery size and would make hedging look artificially bad.
    # Settlement stays feasible because imbalance energy can always top the
    # battery up, at the penalty price, which is the correct economics.
    settled = D.evaluate_commitment(commit.schedule, pv_actual, load, hours, bat, mkt,
                                    terminal_soc=terminal)
    return {
        "policy": policy,
        "status": settled["status"],
        "total_cost": settled["total_cost"],
        "day_ahead_cost": settled["day_ahead_cost"],
        "imbalance_cost": settled["imbalance_cost"],
        "energy_short_mwh": settled["energy_short_mwh"],
        "energy_long_mwh": settled["energy_long_mwh"],
        "committed_import_mwh": float(np.maximum(commit.schedule, 0).sum()),
        "committed_export_mwh": float(np.maximum(-commit.schedule, 0).sum()),
    }


def newsvendor_quantile(market: MarketConfig) -> float:
    """The critical fractile implied by the asymmetric imbalance penalties.

    For a pure newsvendor the cost-optimal commitment quantile is
    ``c_under / (c_under + c_over)``.  The battery and the day-ahead price make
    the real problem richer than a newsvendor, but this number is the right
    first guess for where the optimal commitment sits -- and the empirical
    quantile sweep in experiment 2 is checked against it.
    """
    cu = market.imbalance_penalty_up - market.import_price_offpeak
    co = market.imbalance_penalty_down
    return float(cu / (cu + co))
