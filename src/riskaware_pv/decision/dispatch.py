"""Two-stage stochastic dispatch for a PV + battery prosumer.

This is the module that makes the study a *decision* study rather than a
forecasting exercise.  The economics are the standard European day-ahead
arrangement:

* **Stage 1 (here-and-now, at gate closure).**  Commit an hourly grid exchange
  schedule ``g`` for all 24 hours of the delivery day.  Imports are bought at
  the day-ahead price, exports sold at the feed-in price.  This decision must be
  made from a *forecast*.
* **Stage 2 (recourse, during delivery).**  PV is realised.  The battery can
  absorb part of the mismatch; whatever remains is settled as imbalance -- short
  positions bought at a punitive price, long positions spilled for little.

Because the imbalance penalties are asymmetric (being short costs far more than
being long), the cost-optimal commitment is *not* driven by the mean forecast.
It is driven by a quantile of the predictive distribution determined by the
penalty ratio -- the classic newsvendor result.  A point forecast cannot express
this, which is precisely the gap the probabilistic pipeline fills.

Every problem here is a linear program solved with HiGHS through
``scipy.optimize.linprog``; no commercial solver is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ..config import BatteryConfig, MarketConfig


@dataclass
class DispatchResult:
    status: str
    objective: float
    schedule: np.ndarray          # committed net grid import per hour (MWh)
    import_mwh: np.ndarray
    export_mwh: np.ndarray
    charge: np.ndarray | None = None
    discharge: np.ndarray | None = None
    soc: np.ndarray | None = None
    imbalance_up: np.ndarray | None = None
    imbalance_down: np.ndarray | None = None
    scenario_costs: np.ndarray | None = None
    var: float | None = None

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"


class _Vars:
    """Contiguous variable-block allocator; keeps the LP index algebra honest."""

    def __init__(self) -> None:
        self.n = 0
        self.blocks: dict[str, tuple[int, tuple[int, ...]]] = {}

    def add(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.blocks[name] = (self.n, shape)
        self.n += size
        return idx


def import_price_vector(hours: np.ndarray, market: MarketConfig) -> np.ndarray:
    peak = set(market.peak_hours)
    return np.array([market.import_price_peak if h in peak else market.import_price_offpeak
                     for h in hours], dtype=float)


@dataclass
class DayPrices:
    """Hourly prices for one delivery day, all in EUR/MWh, expressed as *costs*.

    ``import_price``  paid per MWh of committed import (``gp``)
    ``export_price``  received per MWh of committed export (``gm``)
    ``short_price``   paid per MWh short at settlement (``up``)
    ``long_price``    paid per MWh long at settlement (``dn``); negative means
                      the long energy is bought from you, as under single-price
                      settlement

    Case Study 1 used one fixed tariff (:meth:`from_market`).  The thesis needs
    real, time-varying and possibly negative prices (:meth:`single_price`),
    which is why everything below takes vectors.
    """

    import_price: np.ndarray
    export_price: np.ndarray
    short_price: np.ndarray
    long_price: np.ndarray

    def __post_init__(self) -> None:
        for name in ("import_price", "export_price", "short_price", "long_price"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or inf; fill or drop the day first")
            setattr(self, name, arr)

    @classmethod
    def from_market(cls, hours: np.ndarray, market: MarketConfig) -> "DayPrices":
        """The Case Study 1 synthetic tariff (time-of-use import, flat penalties)."""
        T = len(hours)
        return cls(import_price=import_price_vector(hours, market),
                   export_price=np.full(T, market.export_price),
                   short_price=np.full(T, market.imbalance_penalty_up),
                   long_price=np.full(T, market.imbalance_penalty_down))

    @classmethod
    def single_price(cls, day_ahead: np.ndarray, imbalance: np.ndarray) -> "DayPrices":
        """Real market: trade at the day-ahead price, settle deviations at one
        imbalance price (German reBAP).  Short pays it, long receives it."""
        da = np.asarray(day_ahead, dtype=float)
        imb = np.asarray(imbalance, dtype=float)
        return cls(import_price=da, export_price=da.copy(),
                   short_price=imb, long_price=-imb)

    @classmethod
    def dual_price(cls, day_ahead: np.ndarray, penalty_short: float,
                   penalty_long: float) -> "DayPrices":
        """Stylised asymmetric settlement around a real day-ahead price:
        short is bought back at DA + penalty, long is sold at DA - penalty."""
        da = np.asarray(day_ahead, dtype=float)
        return cls(import_price=da, export_price=da.copy(),
                   short_price=da + penalty_short, long_price=-(da - penalty_long))

    @classmethod
    def gate_view(cls, day_ahead: np.ndarray) -> "DayPrices":
        """What a price-taker knows at the gate under single-price settlement.

        The day-ahead price is known (it clears around the gate); the imbalance
        price is not.  With no view on it, its expectation is the day-ahead
        price, so this is :meth:`single_price` with ``imbalance = day_ahead``.
        Use it to *commit*, and settle with the realised imbalance price.
        """
        da = np.asarray(day_ahead, dtype=float)
        return cls.single_price(da, da.copy())

    def check(self, T: int) -> None:
        for name in ("import_price", "export_price", "short_price", "long_price"):
            if getattr(self, name).shape != (T,):
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected ({T},)")

    def speculation_hours(self, tol: float = 1e-9) -> np.ndarray:
        """Hours where a schedule can make money by being *deliberately* off.

        If deviating is cheaper than trading, the optimiser stops forecasting
        and starts speculating on the spread, bounded only by the grid limit.
        That is illegal for a German balance responsible party, and it is never
        a forecasting result.  It is ruled out when, every hour,

        * short is no cheaper than selling: ``short >= export``
        * long earns no more than buying costs: ``import + long >= 0``
        * buying and selling the same hour does not pay: ``import >= export``
        * short and long at once does not pay: ``short + long >= 0``
        """
        bad = ((self.short_price < self.export_price - tol)
               | (self.import_price + self.long_price < -tol)
               | (self.import_price < self.export_price - tol)
               | (self.short_price + self.long_price < -tol))
        return np.flatnonzero(bad)


def _prices(hours: np.ndarray, market: MarketConfig, prices: DayPrices | None) -> DayPrices:
    p = DayPrices.from_market(hours, market) if prices is None else prices
    p.check(len(hours))
    return p


def grid_limits(bat: BatteryConfig, load: np.ndarray, pv: np.ndarray
                ) -> tuple[float, float, float]:
    """Upper bounds on committed import, committed export and imbalance (MWh/h).

    With a fixed positive tariff these never bind: no sensible schedule imports
    more than load plus a full charge, or exports more than PV plus a full
    discharge.  They exist because real prices can be negative, and an LP with
    a negative price on an unbounded variable has no solution at all.
    """
    import_max = bat.grid_import_max_mw
    if import_max is None:
        import_max = float(np.max(load)) + bat.power_mw
    export_max = bat.grid_export_max_mw
    if export_max is None:
        export_max = max(bat.pv_nameplate_mw, float(np.max(pv))) + bat.power_mw
    imbalance_max = (float(np.max(load)) + float(np.max(pv)) + import_max + export_max
                     + 2.0 * bat.power_mw)
    return float(import_max), float(export_max), float(imbalance_max)


def _limits_apply(bat: BatteryConfig, prices: DayPrices | None) -> bool:
    """Bounds are switched on for explicit prices or explicit grid limits.

    They are left off on the plain Case Study 1 path on purpose.  The
    deterministic commitment LP has *tied* optima (several schedules with the
    same planned cost, e.g. when to charge from cheap off-peak import), and any
    change to the bounds lets HiGHS pick a different one.  The planned cost is
    identical, but the settled cost against realised PV is not, so adding even
    non-binding bounds would shift the published Case Study 1 numbers.
    """
    return (prices is not None or bat.grid_import_max_mw is not None
            or bat.grid_export_max_mw is not None)


def _battery_bounds(bat: BatteryConfig, T: int, S: int, v: _Vars,
                    i_c, i_d, i_soc, i_up, i_dn, i_gp, i_gm,
                    extra: int = 0,
                    limits: tuple[float, float, float] | None = None,
                    ) -> list[tuple[float, float | None]]:
    imp_max, exp_max, imb_max = limits if limits is not None else (None, None, None)
    bounds: list[tuple[float, float | None]] = [(None, None)] * (v.n + extra)
    for t in range(T):
        bounds[i_gp[t]] = (0.0, imp_max)
        bounds[i_gm[t]] = (0.0, exp_max)
    for s in range(S):
        for t in range(T):
            bounds[i_c[s, t]] = (0.0, bat.power_mw)
            bounds[i_d[s, t]] = (0.0, bat.power_mw)
            bounds[i_soc[s, t]] = (bat.soc_min * bat.energy_mwh, bat.soc_max * bat.energy_mwh)
            bounds[i_up[s, t]] = (0.0, imb_max)
            bounds[i_dn[s, t]] = (0.0, imb_max)
    return bounds


def _core_constraints(T, S, pv, load, bat, v, i_c, i_d, i_soc, i_up, i_dn, i_gp, i_gm,
                      n_cols, fixed_schedule: np.ndarray | None):
    """Energy balance + state-of-charge dynamics for every scenario."""
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for s in range(S):
        for t in range(T):
            # pv + gp - gm + d - c + up - dn = load
            entries = [(i_d[s, t], 1.0), (i_c[s, t], -1.0),
                       (i_up[s, t], 1.0), (i_dn[s, t], -1.0)]
            b = load[t] - pv[s, t]
            if fixed_schedule is None:
                entries += [(i_gp[t], 1.0), (i_gm[t], -1.0)]
            else:
                b -= fixed_schedule[t]
            for c_, val in entries:
                rows.append(r); cols.append(c_); vals.append(val)
            rhs.append(b)
            r += 1
    for s in range(S):
        for t in range(T):
            # soc_t - soc_{t-1} - eta_c * c_t + d_t / eta_d = 0
            rows += [r, r, r]
            cols += [i_soc[s, t], i_c[s, t], i_d[s, t]]
            vals += [1.0, -bat.eta_charge, 1.0 / bat.eta_discharge]
            if t == 0:
                rhs.append(bat.soc_init * bat.energy_mwh)
            else:
                rows.append(r); cols.append(i_soc[s, t - 1]); vals.append(-1.0)
                rhs.append(0.0)
            r += 1
    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(r, n_cols)).tocsr()
    return A_eq, np.array(rhs, dtype=float)


def solve_commitment(
    pv_scenarios: np.ndarray,
    load: np.ndarray,
    hours: np.ndarray,
    battery: BatteryConfig,
    market: MarketConfig,
    weights: np.ndarray | None = None,
    cvar_weight: float = 0.0,
    cvar_alpha: float | None = None,
    terminal_soc: float | None = None,
    prices: DayPrices | None = None,
    allow_speculation: bool = False,
) -> DispatchResult:
    """Stage-1 commitment under a set of PV scenarios.

    ``cvar_weight`` interpolates between risk-neutral (0 -> minimise expected
    cost) and risk-averse (1 -> minimise CVaR of the recourse cost at level
    ``cvar_alpha``).  Rockafellar & Uryasev's linearisation keeps the problem an
    LP: CVaR = eta + 1/((1-alpha) S) * sum_s max(cost_s - eta, 0).
    """
    pv_scenarios = np.atleast_2d(np.asarray(pv_scenarios, dtype=float))
    S, T = pv_scenarios.shape
    load = np.asarray(load, dtype=float)
    alpha = market.cvar_alpha if cvar_alpha is None else cvar_alpha
    w = np.full(S, 1.0 / S) if weights is None else np.asarray(weights) / np.sum(weights)
    pr = _prices(hours, market, prices)
    bad = pr.speculation_hours()
    if bad.size and not allow_speculation:
        raise ValueError(
            f"commitment prices reward deliberate imbalance in hours {bad.tolist()}; "
            "commit with prices known at the gate (DayPrices.gate_view) and pass the "
            "realised imbalance price only to settlement")

    v = _Vars()
    i_gp = v.add("gp", (T,)); i_gm = v.add("gm", (T,))
    i_c = v.add("c", (S, T)); i_d = v.add("d", (S, T)); i_soc = v.add("soc", (S, T))
    i_up = v.add("up", (S, T)); i_dn = v.add("dn", (S, T))
    use_cvar = cvar_weight > 0.0
    if use_cvar:
        i_eta = v.add("eta", (1,))[0]
        i_z = v.add("z", (S,))
    n_cols = v.n

    limits = (grid_limits(battery, load, pv_scenarios) if _limits_apply(battery, prices)
              else None)
    bounds = _battery_bounds(battery, T, S, v, i_c, i_d, i_soc, i_up, i_dn, i_gp, i_gm,
                             limits=limits)
    if use_cvar:
        bounds[i_eta] = (None, None)
        for s in range(S):
            bounds[i_z[s]] = (0.0, None)

    A_eq, b_eq = _core_constraints(T, S, pv_scenarios, load, battery, v,
                                   i_c, i_d, i_soc, i_up, i_dn, i_gp, i_gm, n_cols, None)

    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    if terminal_soc is not None:
        for s in range(S):  # -soc_T <= -target  (i.e. soc_T >= target)
            rows.append(r); cols.append(i_soc[s, T - 1]); vals.append(-1.0)
            b_ub.append(-terminal_soc * battery.energy_mwh)
            r += 1
    if use_cvar:
        for s in range(S):  # cost_s - eta - z_s <= 0
            for t in range(T):
                rows += [r, r]
                cols += [i_up[s, t], i_dn[s, t]]
                vals += [pr.short_price[t], pr.long_price[t]]
            rows += [r, r]; cols += [i_eta, i_z[s]]; vals += [-1.0, -1.0]
            b_ub.append(0.0)
            r += 1
    A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, n_cols)).tocsr() if r else None
    b_ub_arr = np.array(b_ub, dtype=float) if r else None

    c = np.zeros(n_cols)
    c[i_gp] = pr.import_price
    c[i_gm] = -pr.export_price
    recourse_w = (1.0 - cvar_weight) if use_cvar else 1.0
    for s in range(S):
        c[i_up[s, :]] = recourse_w * w[s] * pr.short_price
        c[i_dn[s, :]] = recourse_w * w[s] * pr.long_price
    if use_cvar:
        c[i_eta] = cvar_weight
        c[i_z] = cvar_weight * w / (1.0 - alpha)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub_arr, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return DispatchResult(status=res.message or "infeasible", objective=np.inf,
                              schedule=np.zeros(T), import_mwh=np.zeros(T),
                              export_mwh=np.zeros(T))
    x = res.x
    gp, gm = x[i_gp], x[i_gm]
    up, dn = x[i_up], x[i_dn]
    scen_cost = (up * pr.short_price + dn * pr.long_price).sum(1)
    return DispatchResult(
        status="optimal", objective=float(res.fun), schedule=gp - gm,
        import_mwh=gp, export_mwh=gm, charge=x[i_c], discharge=x[i_d], soc=x[i_soc],
        imbalance_up=up, imbalance_down=dn, scenario_costs=scen_cost,
        var=float(x[i_eta]) if use_cvar else None,
    )


def evaluate_commitment(
    schedule: np.ndarray,
    pv_actual: np.ndarray,
    load: np.ndarray,
    hours: np.ndarray,
    battery: BatteryConfig,
    market: MarketConfig,
    terminal_soc: float | None = None,
    prices: DayPrices | None = None,
    settle_prices: DayPrices | None = None,
) -> dict:
    """Stage-2 only: settle a *fixed* commitment against the realised PV.

    This is the honest scoring step -- the schedule was chosen without knowing
    ``pv_actual``, and the operator now pays for whatever mismatch remains after
    the battery has done its best.

    ``prices`` are what the operator knows and steers the battery by.
    ``settle_prices``, if given, are the realised imbalance prices the
    resulting short/long energy is finally billed at.  Keeping them apart stops
    the battery from trading on an imbalance price nobody knows in advance.
    """
    schedule = np.asarray(schedule, dtype=float)
    pv = np.atleast_2d(np.asarray(pv_actual, dtype=float))
    T = len(schedule)
    S = 1
    pr = _prices(hours, market, prices)

    v = _Vars()
    i_gp = v.add("gp", (T,)); i_gm = v.add("gm", (T,))
    i_c = v.add("c", (S, T)); i_d = v.add("d", (S, T)); i_soc = v.add("soc", (S, T))
    i_up = v.add("up", (S, T)); i_dn = v.add("dn", (S, T))
    n_cols = v.n

    limits = None
    if _limits_apply(battery, prices) or settle_prices is not None:
        imp_max, exp_max, imb_max = grid_limits(battery, load, pv)
        # the committed schedule itself may sit at the grid limit, so the
        # imbalance bound must also cover the schedule's own size
        limits = (imp_max, exp_max, imb_max + float(np.max(np.abs(schedule))))
    bounds = _battery_bounds(battery, T, S, v, i_c, i_d, i_soc, i_up, i_dn, i_gp, i_gm,
                             limits=limits)
    for t in range(T):  # stage-1 decision is frozen
        bounds[i_gp[t]] = (0.0, 0.0)
        bounds[i_gm[t]] = (0.0, 0.0)

    A_eq, b_eq = _core_constraints(T, S, pv, load, battery, v, i_c, i_d, i_soc,
                                   i_up, i_dn, i_gp, i_gm, n_cols, schedule)
    A_ub = b_ub = None
    if terminal_soc is not None:
        A_ub = sparse.coo_matrix(
            ([-1.0], ([0], [i_soc[0, T - 1]])), shape=(1, n_cols)).tocsr()
        b_ub = np.array([-terminal_soc * battery.energy_mwh])

    c = np.zeros(n_cols)
    c[i_up[0, :]] = pr.short_price
    c[i_dn[0, :]] = pr.long_price
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")

    day_ahead_cost = float(np.sum(np.maximum(schedule, 0) * pr.import_price)
                           - np.sum(np.maximum(-schedule, 0) * pr.export_price))
    if not res.success:
        return {"status": "infeasible", "total_cost": np.inf,
                "day_ahead_cost": day_ahead_cost, "imbalance_cost": np.inf,
                "energy_short_mwh": np.nan, "energy_long_mwh": np.nan}
    x = res.x
    up, dn = x[i_up[0, :]], x[i_dn[0, :]]
    if settle_prices is None:
        imbalance_cost = float(res.fun)
    else:
        settle_prices.check(T)
        imbalance_cost = float(np.sum(up * settle_prices.short_price
                                      + dn * settle_prices.long_price))
    return {
        "status": "optimal",
        "day_ahead_cost": day_ahead_cost,
        "imbalance_cost": imbalance_cost,
        "total_cost": day_ahead_cost + imbalance_cost,
        "energy_short_mwh": float(up.sum()),
        "energy_long_mwh": float(dn.sum()),
        "soc": x[i_soc[0, :]],
        "charge": x[i_c[0, :]],
        "discharge": x[i_d[0, :]],
    }


def oracle_cost(pv_actual: np.ndarray, load: np.ndarray, hours: np.ndarray,
                battery: BatteryConfig, market: MarketConfig,
                terminal_soc: float | None = None,
                prices: DayPrices | None = None) -> float:
    """Perfect-foresight cost: the unattainable lower bound that defines regret."""
    res = solve_commitment(np.atleast_2d(pv_actual), load, hours, battery, market,
                           terminal_soc=terminal_soc, prices=prices)
    return float(res.objective) if res.feasible else float("inf")
