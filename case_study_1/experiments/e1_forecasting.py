"""Experiment 1 - day-ahead probabilistic forecasting.

Answers RQ1: how much skill is available without NWP input, and does conformal
calibration deliver valid coverage under seasonal non-stationarity?

Outputs
-------
results/e1_scores.csv          per-fold, per-model, per-subset scores
results/e1_leaderboard.csv     fold-averaged leaderboard
results/e1_predictions.parquet per-hour forecasts (consumed by e2 and e4)
figures/01..06         data overview, leaderboard, reliability,
                               coverage-vs-width, fan chart, error decomposition
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from riskaware_pv.config import Config, RESULTS
from riskaware_pv.evaluation import metrics as M
from riskaware_pv.evaluation.backtest import make_folds, run_backtest, summarise
from riskaware_pv.explain import error_decomposition as ED
from riskaware_pv.pipeline import (load_design_matrix, load_raw_frame,
                                   model_registry, quantile_columns, save_table)
from riskaware_pv.viz import plots


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="fewer trees / epochs")
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--folds", type=int, default=None, help="use only the last N folds")
    args = ap.parse_args(argv)

    cfg = Config.load()
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    print("== loading data ==")
    frame = load_raw_frame(cfg)
    X = load_design_matrix(cfg)
    print(f"   design matrix {X.shape}, {X['origin'].nunique()} delivery days, "
          f"{X.index.min().date()} -> {X.index.max().date()}")

    folds = make_folds(X.index, cfg, n_folds=args.folds)
    print(f"== backtest over {len(folds)} rolling folds ==")
    registry = model_registry(cfg, include_lstm=not args.no_lstm, fast=args.fast)
    print(f"   models: {', '.join(registry)}")

    scores, preds = run_backtest(X, registry, cfg, folds=folds, conformalise=True)
    save_table(scores, "e1_scores.csv")
    save_table(preds, "e1_predictions.parquet")

    board = summarise(scores, subset="daylight")
    save_table(board, "e1_leaderboard.csv")
    print("\n== leaderboard (daylight hours, fold-averaged) ==")
    cols = [c for c in ("crps", "pinball", "mae", "rmse", "picp90", "pinaw90",
                        "winkler90", "mae_skill_vs_persistence") if c in board.columns]
    print(board[cols].round(4).to_string())

    # ---------------- figures ------------------------------------------------
    print("\n== figures ==")
    print(" ", plots.plot_data_overview(frame, X).name)
    print(" ", plots.plot_leaderboard(board).name)
    print(" ", plots.plot_coverage_vs_width(scores).name)

    qs = np.asarray(cfg.forecast.quantiles)
    qcols = quantile_columns(cfg)
    day = preds["daylight"].to_numpy() if "daylight" in preds else slice(None)
    curves = {}
    for (model, variant), grp in preds.groupby(["model", "variant"]):
        if not set(qcols) <= set(grp.columns):
            continue
        g = grp[grp["daylight"]] if "daylight" in grp else grp
        if len(g) < 100:
            continue
        curves[f"{model} [{variant}]"] = M.reliability_curve(
            g["y"].to_numpy(), g[qcols].to_numpy(), qs)
    keep = [k for k in curves if any(t in k for t in
            ("gbm_quantile [", "gbm_quantile [cqr", "lstm", "climatology_quantiles [raw"))]
    print(" ", plots.plot_reliability({k: curves[k] for k in keep[:6]}).name)

    best = board.index[0]
    best_preds = preds[(preds["model"] == best[0]) & (preds["variant"] == best[1])]
    # Choose the most meteorologically varied stretch, so the fan chart shows
    # how interval width responds to conditions rather than a run of clear days.
    start = plots.most_variable_window(best_preds, days=5)
    print(" ", plots.plot_fan_chart(best_preds, qs, start, days=5,
                                    title=f"{best[0]} [{best[1]}] - 5 consecutive days").name)

    by_h = ED.by_horizon(best_preds)
    by_hour = ED.by_hour_of_day(best_preds)
    by_reg = ED.by_regime(best_preds, X)
    conc = ED.error_concentration(best_preds)
    print(" ", plots.plot_error_decomposition(by_h, by_hour, by_reg, conc).name)

    save_table(by_h, "e1_error_by_horizon.csv")
    save_table(by_reg, "e1_error_by_regime.csv")
    save_table(ED.worst_days(best_preds, 15), "e1_worst_days.csv")
    save_table(conc.to_frame("share"), "e1_error_concentration.csv")

    print(f"\n== done in {time.time() - t0:.0f}s | best model: {best[0]} [{best[1]}] ==")
    print(f"   error concentration: worst 10% of daylight hours carry "
          f"{conc['top_10pct_share']:.0%} of all absolute error")


if __name__ == "__main__":
    main()
