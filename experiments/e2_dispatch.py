"""Experiment 2 - does a better forecast buy a better decision?

Answers RQ2.  Each delivery day in the backtest is settled through the
two-stage dispatch problem under five commitment policies, and the cost is
compared against the perfect-foresight oracle to give a *regret* in euros.

The central claim under test is that ranking models by CRPS is not the same as
ranking them by money, and that the gap between a point forecast and the full
predictive distribution is worth more than the gap between two good point
forecasts.

Outputs
-------
results/e2_daily.csv         per-day, per-policy cost breakdown
results/e2_summary.csv       policy-level summary with regret
results/e2_quantile_sweep.csv  regret as a function of commitment quantile
results/e2_by_model.csv      regret for each forecasting model
figures/07, 08, 11
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from riskaware_pv.config import Config
from riskaware_pv.decision import policies as P
from riskaware_pv.evaluation import metrics as M
from riskaware_pv.pipeline import quantile_columns, save_table
from riskaware_pv.config import RESULTS
from riskaware_pv.viz import plots

POLICIES = ("oracle", "point", "quantile", "stochastic", "cvar")


def _day_frames(preds: pd.DataFrame, cfg: Config, max_days: int | None,
                seed: int) -> list[pd.DataFrame]:
    """Split a model's predictions into complete delivery days."""
    days = [g for _, g in preds.groupby(preds.index.normalize()) if len(g) == cfg.forecast.horizon]
    if max_days is not None and len(days) > max_days:
        # Systematic thinning keeps the seasonal coverage of the sample intact,
        # which random subsampling of a strongly seasonal series would not.
        step = len(days) / max_days
        days = [days[int(i * step)] for i in range(max_days)]
    return days


def run_for_model(preds: pd.DataFrame, cfg: Config, max_days: int | None = None,
                  policies=POLICIES, verbose: bool = True) -> pd.DataFrame:
    qcols = quantile_columns(cfg)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    days = _day_frames(preds, cfg, max_days, cfg.seed)
    for i, day in enumerate(days):
        day = day.sort_index()
        q_pred = day[qcols].to_numpy()
        point = day["point"].to_numpy()
        actual = day["y"].to_numpy()
        inputs = P.build_day_inputs(day, cfg)
        for pol in policies:
            out = P.run_policy(pol, inputs, q_pred, point, actual, cfg, rng=rng)
            out["date"] = day.index[0].normalize()
            rows.append(out)
        if verbose and (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(days)} days")
    df = pd.DataFrame(rows)
    oracle = df[df["policy"] == "oracle"].set_index("date")["total_cost"]
    df["regret"] = df["total_cost"] - df["date"].map(oracle)
    return df


def summarise_policies(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("policy")
    out = pd.DataFrame({
        "mean_cost_eur": g["total_cost"].mean(),
        "mean_regret_eur": g["regret"].mean(),
        "median_regret_eur": g["regret"].median(),
        "p95_regret_eur": g["regret"].quantile(0.95),
        "mean_day_ahead_eur": g["day_ahead_cost"].mean(),
        "mean_imbalance_eur": g["imbalance_cost"].mean(),
        "mean_short_mwh": g["energy_short_mwh"].mean(),
        "mean_long_mwh": g["energy_long_mwh"].mean(),
        "n_days": g.size(),
    })
    base = out.loc["point", "mean_regret_eur"] if "point" in out.index else np.nan
    out["regret_vs_point_pct"] = 100.0 * (out["mean_regret_eur"] - base) / base
    return out.sort_values("mean_regret_eur")


def quantile_sweep(preds: pd.DataFrame, cfg: Config, max_days: int | None,
                   levels=(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
                   ) -> pd.DataFrame:
    """Regret as a function of the fixed commitment quantile.

    The minimum of this curve is the empirical answer to "how conservative
    should the schedule be?", and it is compared with the newsvendor fractile
    implied by the penalty asymmetry.
    """
    qcols = quantile_columns(cfg)
    days = _day_frames(preds, cfg, max_days, cfg.seed)
    rows = []
    for level in levels:
        costs, oracles = [], []
        for day in days:
            day = day.sort_index()
            inputs = P.build_day_inputs(day, cfg)
            q_pred = day[qcols].to_numpy()
            point, actual = day["point"].to_numpy(), day["y"].to_numpy()
            o = P.run_policy("oracle", inputs, q_pred, point, actual, cfg)
            r = P.run_policy("quantile", inputs, q_pred, point, actual, cfg,
                             fixed_quantile=level)
            if np.isfinite(r["total_cost"]) and np.isfinite(o["total_cost"]):
                costs.append(r["total_cost"]); oracles.append(o["total_cost"])
        rows.append({"quantile": level,
                     "mean_regret_eur": float(np.mean(costs) - np.mean(oracles)),
                     "mean_cost_eur": float(np.mean(costs))})
    return pd.DataFrame(rows)


def sensitivity_study(preds: pd.DataFrame, cfg: Config, max_days: int,
                      penalties=(150.0, 300.0, 600.0, 1200.0),
                      batteries=(0.0, 2.0, 4.0, 8.0)) -> pd.DataFrame:
    """When is probabilistic forecasting actually worth the trouble?

    The headline single-number comparison of "stochastic vs point" depends
    entirely on two quantities the operator does not choose freely: how badly
    imbalance is penalised, and how much storage flexibility is available to
    absorb a forecast error after the fact.

    The prior hypothesis was that a battery *substitutes* for a good forecast --
    it buys ex post the same insurance a conservative commitment buys ex ante --
    so the value of the predictive distribution should fall as storage grows.

    The grid refutes that.  The advantage of the stochastic commitment *rises*
    with storage (see results/e2_sensitivity.csv).  The reason is that the
    second-stage recourse is what actually exploits a scenario set: with no
    battery there is nothing to re-optimise once PV is realised, so knowing the
    distribution buys little beyond a shifted commitment, whereas a large
    battery gives the stochastic program a lever the point forecast never asks
    it to pull.  Storage and probabilistic forecasting are complements, not
    substitutes.  The hypothesis is left documented here because a refuted
    prediction is a result, not an embarrassment.
    """
    import copy
    rows = []
    for pen in penalties:
        for e_mwh in batteries:
            c = copy.deepcopy(cfg)
            c.market.imbalance_penalty_up = pen
            c.battery.energy_mwh = e_mwh
            # power rating scales with energy at a fixed 4-hour duration;
            # a zero-energy battery must also have zero power.
            c.battery.power_mw = e_mwh / 4.0
            d = run_for_model(preds, c, max_days,
                              policies=("oracle", "point", "stochastic"), verbose=False)
            s = summarise_policies(d)
            if not {"point", "stochastic"} <= set(s.index):
                continue
            pt, st = s.loc["point", "mean_regret_eur"], s.loc["stochastic", "mean_regret_eur"]
            rows.append({
                "penalty_up": pen, "battery_mwh": e_mwh,
                "regret_point": pt, "regret_stochastic": st,
                "advantage_eur": pt - st,
                "advantage_pct": 100.0 * (pt - st) / pt if pt > 0 else np.nan,
            })
            print(f"   penalty {pen:>6.0f} EUR/MWh, battery {e_mwh:>4.1f} MWh -> "
                  f"point {pt:7.2f}  stochastic {st:7.2f}  "
                  f"advantage {rows[-1]['advantage_pct']:+6.1f}%")
    return pd.DataFrame(rows)


def plot_sensitivity(sens: pd.DataFrame):
    import matplotlib.pyplot as plt
    from riskaware_pv.viz.plots import _save
    piv = sens.pivot(index="penalty_up", columns="battery_mwh", values="advantage_pct")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
    ax.set_xlabel("battery energy (MWh)"); ax.set_ylabel("imbalance penalty (EUR/MWh)")
    ax.set_title("Advantage of stochastic over point commitment (%)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label="% regret reduction")
    ax.grid(False)

    ax = axes[1]
    for e in sorted(sens["battery_mwh"].unique()):
        sub = sens[sens["battery_mwh"] == e].sort_values("penalty_up")
        ax.plot(sub["penalty_up"], sub["advantage_pct"], marker="o", ms=4,
                label=f"{e:g} MWh")
    ax.axhline(0, color="#999", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("imbalance penalty, short (EUR/MWh, log)")
    ax.set_ylabel("regret reduction vs point forecast (%)")
    # Title states what the data shows, not the hypothesis it refuted: at high
    # penalties the advantage *grows* with storage, because recourse is what
    # makes a scenario set worth having.
    ax.set_title("At high penalties, storage and forecasting are complements")
    ax.legend(fontsize=8, title="battery", title_fontsize=8)
    fig.suptitle("When does the predictive distribution pay for itself?", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, "13_sensitivity.png").name


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-days", type=int, default=200,
                    help="delivery days to settle per model (thinned across seasons)")
    ap.add_argument("--model", default=None, help="model to use for the policy study")
    ap.add_argument("--sensitivity", action="store_true",
                    help="sweep imbalance penalty x battery size")
    ap.add_argument("--all-models", action="store_true",
                    help="also compute regret for every model (forecast-vs-value figure)")
    args = ap.parse_args(argv)

    cfg = Config.load()
    t0 = time.time()
    preds = pd.read_parquet(RESULTS / "e1_predictions.parquet")
    preds.index = pd.DatetimeIndex(preds.index)
    board = pd.read_csv(RESULTS / "e1_leaderboard.csv", index_col=[0, 1])

    if args.model:
        model, variant = args.model, "cqr"
    else:
        model, variant = board.index[0]
    print(f"== policy study using {model} [{variant}] ==")
    sub = preds[(preds["model"] == model) & (preds["variant"] == variant)].sort_index()

    daily = run_for_model(sub, cfg, args.max_days)
    save_table(daily, "e2_daily.csv")
    summary = summarise_policies(daily)
    save_table(summary, "e2_summary.csv")
    print("\n== policy comparison (EUR per delivery day) ==")
    print(summary.round(2).to_string())

    print("\n== commitment-quantile sweep ==")
    sweep = quantile_sweep(sub, cfg, min(args.max_days, 120))
    save_table(sweep, "e2_quantile_sweep.csv")
    crit = P.newsvendor_quantile(cfg.market)
    print(sweep.round(2).to_string(index=False))
    print(f"   newsvendor critical fractile = {crit:.3f}; "
          f"empirical optimum = {sweep.loc[sweep['mean_regret_eur'].idxmin(), 'quantile']:.2f}")

    print("\n== figures ==")
    print(" ", plots.plot_dispatch_results(daily, summary).name)
    print(" ", plots.plot_quantile_sweep(sweep, crit).name)

    if args.sensitivity:
        print("\n== sensitivity: penalty x storage ==")
        sens = sensitivity_study(sub, cfg, min(args.max_days, 90))
        save_table(sens, "e2_sensitivity.csv")
        print(" ", plot_sensitivity(sens))

    if args.all_models:
        print("\n== regret for every forecasting model (stochastic policy) ==")
        rows = []
        qcols = quantile_columns(cfg)
        qs = np.asarray(cfg.forecast.quantiles)
        for (m, v), grp in preds.groupby(["model", "variant"]):
            if not set(qcols) <= set(grp.columns):
                continue
            d = run_for_model(grp.sort_index(), cfg, min(args.max_days, 120),
                              policies=("oracle", "stochastic"), verbose=False)
            s = summarise_policies(d)
            day = grp[grp["daylight"]] if "daylight" in grp else grp
            rows.append({
                "label": f"{m}[{v}]", "model": m, "variant": v,
                "crps": M.crps_from_quantiles(day["y"].to_numpy(), day[qcols].to_numpy(), qs),
                "mae": M.mae(day["y"].to_numpy(), day["point"].to_numpy()),
                "mean_regret_eur": s.loc["stochastic", "mean_regret_eur"],
            })
            print(f"   {m}[{v}]: CRPS {rows[-1]['crps']:.4f} -> regret "
                  f"EUR {rows[-1]['mean_regret_eur']:.2f}/day")
        joint = pd.DataFrame(rows)
        save_table(joint, "e2_by_model.csv")
        print(" ", plots.plot_forecast_vs_decision(joint).name)
        r = joint["crps"].corr(joint["mean_regret_eur"])
        print(f"   correlation between CRPS and dispatch regret: r = {r:.3f}")

    print(f"\n== done in {time.time() - t0:.0f}s ==")


if __name__ == "__main__":
    main()
