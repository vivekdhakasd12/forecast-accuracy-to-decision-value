"""Experiment 3 - concept drift and retraining policy.

Answers RQ3.  Three retraining policies are simulated day-by-day over the test
period at an *equal retraining budget*, and a separate adaptive-conformal study
asks whether online interval calibration can hold nominal coverage when the
split-conformal guarantee is broken by seasonal shift.

Two questions, deliberately separated:

* **Point accuracy.**  Does drift-triggered retraining beat calendar retraining
  for the same number of refits?
* **Interval validity.**  Experiment 1 showed split-conformal intervals still
  under-cover over a 12-month test period.  Adaptive conformal inference updates
  the level online; does it recover nominal coverage, and at what cost in width?

Outputs
-------
results/e3_policy_summary.csv, e3_feature_drift.csv, e3_aci.csv
figures/09, 12
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from riskaware_pv.config import Config, RESULTS
from riskaware_pv.drift import detectors as DD
from riskaware_pv.drift.policies import rolling_scores, score_policy, simulate_policy
from riskaware_pv.evaluation import metrics as M
from riskaware_pv.models.conformal import AdaptiveConformal
from riskaware_pv.pipeline import load_design_matrix, model_registry, save_table
from riskaware_pv.viz import plots


def feature_drift_report(X: pd.DataFrame, cfg: Config, split: pd.Timestamp) -> pd.DataFrame:
    """PSI/KS between the early and late halves of the sample.

    Run on the *inputs*, so it would fire even if labels were delayed -- the
    usual operational reality.
    """
    from riskaware_pv.data.features import feature_columns
    cols = [c for c in feature_columns(X) if c not in ("origin",)]
    ref, cur = X.loc[X.index < split], X.loc[X.index >= split]
    rep = DD.feature_drift(ref, cur, cols, cfg.drift.psi_threshold, cfg.drift.psi_bins)
    out = pd.DataFrame({"psi": rep.psi, "ks_stat": rep.ks_stat,
                        "ks_pvalue": rep.ks_pvalue}).sort_values("psi", ascending=False)
    out["flagged"] = out.index.isin(rep.flagged)
    return out


def adaptive_conformal_study(X: pd.DataFrame, cfg: Config, test_start: pd.Timestamp,
                             gammas=(0.0, 0.005, 0.02, 0.05)) -> pd.DataFrame:
    """Split-conformal (gamma = 0) versus adaptive conformal at several rates."""
    from riskaware_pv.models.gbm import GBMQuantile
    cal_days = cfg.backtest.calibration_days
    hist = X.loc[X.index < test_start]
    cut = test_start - pd.Timedelta(days=cal_days)
    X_tr, X_cal = hist.loc[hist.index < cut], hist.loc[hist.index >= cut]
    X_te = X.loc[X.index >= test_start]
    # Daylight hours only: coverage of a zero-valued night hour is meaningless.
    X_te = X_te[X_te["cos_zenith"] > 1e-3]

    model = GBMQuantile(quantiles=cfg.forecast.quantiles, params=dict(n_estimators=400))
    model.fit(X_tr, X_tr["y"].to_numpy())

    rows = []
    alpha = 0.10
    y_cal = X_cal["y"].to_numpy()
    y_te = X_te["y"].to_numpy()
    for gamma in gammas:
        # gamma = 0 is ordinary split conformal on the same score, so the only
        # thing that varies across rows is the online adaptation itself.
        aci = AdaptiveConformal(model, alpha=alpha, gamma=gamma).seed(X_cal, y_cal)
        traj = aci.run(X_te, y_te)
        label = ("split conformal (static, gamma=0)" if gamma == 0.0
                 else f"adaptive conformal (gamma={gamma})")
        rows.append({
            "method": label, "gamma": gamma,
            "coverage": float(1.0 - traj["miss"].mean()),
            "target_coverage": 1.0 - alpha,
            "mean_width": float((traj["upper"] - traj["lower"]).mean()),
            "final_alpha_t": float(traj["alpha_t"].iloc[-1]),
            "worst_month_coverage": float(
                (1.0 - traj["miss"].groupby(traj.index.to_period("M")).mean()).min()),
        })
        print(f"   {label:38s} coverage {rows[-1]['coverage']:.3f} "
              f"(worst month {rows[-1]['worst_month_coverage']:.3f}), "
              f"width {rows[-1]['mean_width']:.4f}")
        rows[-1]["_traj"] = traj
    return pd.DataFrame(rows)


def plot_aci(aci: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt
    from riskaware_pv.viz.plots import _save
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    for _, r in aci.iterrows():
        traj = r["_traj"]
        cov = (1.0 - traj["miss"]).rolling(24 * 30, min_periods=200).mean()
        ax.plot(traj.index, cov, lw=1.4, label=r["method"])
    ax.axhline(0.90, color="#e45756", ls="--", lw=1.2, label="nominal 90%")
    ax.set_ylabel("30-day rolling coverage"); ax.set_title("Coverage over the test year")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.scatter(aci["mean_width"], aci["coverage"], s=80, c="#4c78a8")
    for _, r in aci.iterrows():
        ax.annotate(f"g={r['gamma']}", (r["mean_width"], r["coverage"]), fontsize=8,
                    xytext=(5, 3), textcoords="offset points")
    ax.axhline(0.90, color="#e45756", ls="--", lw=1.2)
    ax.set_xlabel("mean interval width"); ax.set_ylabel("realised coverage")
    ax.set_title("The price of validity")
    fig.suptitle("Adaptive conformal inference under seasonal drift", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, "12_adaptive_conformal.png").name


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gbm_quantile")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load()
    t0 = time.time()
    X = load_design_matrix(cfg)
    registry = model_registry(cfg, include_lstm=False, fast=args.fast)
    factory = registry[args.model]

    test_start = X.index.min() + pd.Timedelta(
        days=cfg.backtest.min_train_days + cfg.backtest.calibration_days)
    test_start = max(test_start, X.index.max() - pd.Timedelta(days=730))
    print(f"== drift study: test period {test_start.date()} -> {X.index.max().date()} ==")

    print("\n== feature drift (PSI / KS, early vs late sample) ==")
    split = X.index.min() + (X.index.max() - X.index.min()) / 2
    fd = feature_drift_report(X, cfg, split)
    save_table(fd, "e3_feature_drift.csv")
    print(fd.head(10).round(4).to_string())
    print(f"   {int(fd['flagged'].sum())} of {len(fd)} features exceed PSI > "
          f"{cfg.drift.psi_threshold}")

    print("\n== retraining policies at equal budget ==")
    budget = args.budget if args.budget is not None else cfg.drift.retrain_budget
    trajectories, logs, rows = {}, {}, []
    for policy in ("static", "periodic", "drift"):
        print(f"  -- {policy} --")
        preds, log = simulate_policy(X, factory, cfg, policy=policy,
                                     test_start=test_start, budget=budget, verbose=True)
        s = score_policy(preds, cfg)
        s.update({"policy": policy, "n_refits": log.n_refits,
                  "n_triggers": len(log.triggers)})
        rows.append(s)
        trajectories[policy] = rolling_scores(preds)
        logs[policy] = log
        # Persist the trajectory so figures can be regenerated without refitting.
        save_table(trajectories[policy], f"e3_trajectory_{policy}.csv")
        save_table(pd.DataFrame({"refit": log.refits}), f"e3_refits_{policy}.csv")
        print(f"     MAE {s['mae']:.4f}  CRPS {s['crps']:.4f}  PICP90 {s.get('picp90', np.nan):.3f}"
              f"  refits {log.n_refits}  triggers {len(log.triggers)}")

    summary = pd.DataFrame(rows).set_index("policy")
    save_table(summary, "e3_policy_summary.csv")
    print("\n== policy summary ==")
    cols = [c for c in ("mae", "rmse", "crps", "picp90", "n_refits", "n_triggers")
            if c in summary.columns]
    print(summary[cols].round(4).to_string())

    print("\n== adaptive conformal inference ==")
    aci = adaptive_conformal_study(X, cfg, test_start)
    save_table(aci.drop(columns=["_traj"]), "e3_aci.csv")

    print("\n== figures ==")
    print(" ", plots.plot_drift(trajectories, logs, summary).name)
    print(" ", plot_aci(aci))
    print(f"\n== done in {time.time() - t0:.0f}s ==")


if __name__ == "__main__":
    main()
