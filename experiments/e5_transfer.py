"""Experiment 5 - a small-data strategy for a new site.

Answers the third novelty in the brief. A newly commissioned site has days of
history, not years, but still has to submit a day-ahead schedule. This experiment
asks how much local history is needed before training locally beats borrowing a
model trained on a different country, and whether a cheap correction closes the
gap sooner.

Source: Germany. Targets: Switzerland, Denmark and Great Britain, spanning
9 degrees of latitude and three quite different climates.

Outputs
-------
results/e5_transfer.csv       one row per (site, strategy, budget)
results/e5_crossover.csv      local-beats-transfer crossover per site
figures/14, 15
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from functools import partial

from riskaware_pv.config import Config
from riskaware_pv.data.features import feature_columns
from riskaware_pv.data.opsd import SITES
from riskaware_pv.evaluation import metrics as M
from riskaware_pv.pipeline import (load_design_matrix_for, model_registry,
                                   save_table)
from riskaware_pv.transfer import STRATEGIES, TransferForecaster, local_slice
from riskaware_pv.viz import plots

BUDGETS = (7, 14, 30, 60, 120, 240, 480)
TARGETS = ("CH", "DK", "GB_GBN")
TEST_DAYS = 60      # length of each held-out evaluation window
N_FOLDS = 5         # evaluation windows per site, spread across the series


def evaluate(model, X_te: pd.DataFrame, quantiles) -> dict:
    """Daylight-only scores on the held-out local test period."""
    y = X_te["y"].to_numpy()
    day = X_te["cos_zenith"].to_numpy() > 1e-3
    point = model.predict(X_te)
    try:
        qp = model.predict_quantiles(X_te, quantiles)
    except NotImplementedError:
        qp = None
    return M.evaluate_all(y[day], point[day], qp[day] if qp is not None else None,
                          quantiles, scale=float(np.mean(y[day]) + 1e-9))


def common_schema(frames: list[pd.DataFrame]) -> list[str]:
    """Features present at every site.

    Germany reports a day-ahead price and the other zones in this package do
    not, so a source model fitted on the full German block asks the target for
    columns that do not exist there. Transfer requires one schema, and the
    intersection is the only one every site can honour.
    """
    common = set(feature_columns(frames[0]))
    for f in frames[1:]:
        common &= set(feature_columns(f))
    return sorted(common)


def fold_origins(X: pd.DataFrame, n_folds: int, test_days: int,
                 max_budget: int) -> list[pd.Timestamp]:
    """Evenly spaced test-window starts, each with room for the largest budget.

    A single window makes the result a lottery over which season the short local
    slice happened to land in: a 30-day window drawn from March and evaluated on
    a summer test period is not merely small, it is seasonally biased, and the
    first version of this experiment produced a wildly non-monotonic curve for
    exactly that reason. Repeating over several origins turns that lottery into
    a mean with a spread.
    """
    first = X.index.min().normalize() + pd.Timedelta(days=max_budget)
    last = X.index.max().normalize() - pd.Timedelta(days=test_days)
    if last <= first:
        return [last]
    span = (last - first).days
    return [first + pd.Timedelta(days=int(round(i * span / max(n_folds - 1, 1))))
            for i in range(n_folds)]


def run_site(country: str, X_src: pd.DataFrame, source_model, cfg: Config,
             base_name: str, schema: list[str], X: pd.DataFrame | None = None,
             verbose: bool = True) -> list[dict]:
    quantiles = np.asarray(cfg.forecast.quantiles, dtype=float)
    reg = model_registry(cfg, include_lstm=False)[base_name]
    factory = (partial(reg, feature_subset=schema, alpha="cv") if base_name == "ridge"
               else partial(reg, feature_subset=schema))

    X = load_design_matrix_for(country, cfg) if X is None else X
    origins = fold_origins(X, N_FOLDS, TEST_DAYS, max(BUDGETS))
    if verbose:
        print(f"  {country} ({SITES[country]['name']}, {SITES[country]['latitude']:.1f}N): "
              f"{len(X)} rows, {len(origins)} folds "
              f"({origins[0].date()} .. {origins[-1].date()})")

    rows = []
    for fold, test_start in enumerate(origins):
        X_te = X.loc[(X.index >= test_start) &
                     (X.index < test_start + pd.Timedelta(days=TEST_DAYS))]
        if len(X_te) < 200:
            continue
        for strategy in STRATEGIES:
            for budget in BUDGETS:
                if strategy == "zero_shot" and budget != BUDGETS[0]:
                    continue
                X_loc = local_slice(X, test_start, budget)
                if strategy != "zero_shot" and len(X_loc) < 48:
                    continue
                model = TransferForecaster(
                    strategy, source=source_model, local_factory=factory,
                    source_frame=X_src if strategy == "pooled" else None,
                )
                try:
                    model.fit(X_loc, X_loc["y"].to_numpy())
                    sc = evaluate(model, X_te, quantiles)
                except Exception as exc:  # pragma: no cover
                    print(f"     ! {strategy}/{budget}d fold {fold}: {type(exc).__name__}")
                    continue
                rows.append({"site": country, "site_name": SITES[country]["name"],
                             "latitude": SITES[country]["latitude"], "fold": fold,
                             "strategy": strategy, "budget_days": budget,
                             "n_train": len(X_loc), "base": base_name, **sc})

    if verbose:
        agg = (pd.DataFrame(rows).groupby(["strategy", "budget_days"])["mae"]
               .mean().unstack())
        print(agg.round(4).to_string())
    return rows


def crossover_table(df: pd.DataFrame) -> pd.DataFrame:
    """Smallest budget from which local-only training *stays* ahead of zero-shot.

    Not the first budget that happens to beat it. The local curve is noisy at
    short budgets -- a fortnight of Swiss data drawn from the right fortnight can
    beat the transferred model once and lose to it at thirty and sixty days --
    and reporting that first crossing would overstate how little history a new
    site needs. The reported figure is the smallest budget at which local wins
    and keeps winning at every larger budget.
    """
    out = []
    for site, g in df.groupby("site"):
        zs = g[g["strategy"] == "zero_shot"]["mae"]
        if zs.empty:
            continue
        zero = float(zs.iloc[0])
        loc = g[g["strategy"] == "local"].sort_values("budget_days")
        budgets = loc["budget_days"].to_numpy()
        wins = (loc["mae"].to_numpy() <= zero)
        stable = [budgets[i] for i in range(len(budgets)) if wins[i:].all()]
        pool = g[g["strategy"] == "pooled"].sort_values("budget_days")

        def at(frame, days):
            hit = frame[frame["budget_days"] == days]["mae"]
            return float(hit.iloc[0]) if len(hit) else np.nan

        out.append({
            "site": site,
            "site_name": g["site_name"].iloc[0],
            "latitude": g["latitude"].iloc[0],
            "zero_shot_mae": zero,
            "local_mae_7d": at(loc, 7),
            "local_mae_480d": at(loc, 480),
            "pooled_mae_7d": at(pool, 7),
            "crossover_days": int(min(stable)) if stable else np.nan,
            "best_finetune_mae": float(g[g["strategy"] == "finetune"]["mae"].min()),
            "best_pooled_mae": float(pool["mae"].min()) if len(pool) else np.nan,
        })
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="ridge", help="base learner for every strategy")
    ap.add_argument("--also-gbm", action="store_true",
                    help="repeat one site with the boosted model")
    args = ap.parse_args(argv)

    cfg = Config.load()
    t0 = time.time()

    print("== aligning the feature schema across sites ==")
    frames = {c: load_design_matrix_for(c, cfg) for c in ("DE", *TARGETS)}
    schema = common_schema(list(frames.values()))
    dropped = sorted(set(feature_columns(frames["DE"])) - set(schema))
    print(f"   {len(schema)} shared features; dropped from the German block: {dropped}")

    print("\n== source model: Germany ==")
    # The source is trained on the *quantile*-normalised German series so that
    # source and targets share one definition of the target variable; Germany is
    # the only zone whose reported capacity is coherent with its generation, so
    # capacity normalisation cannot be applied across sites.
    X_src = frames["DE"]
    src_factory = partial(model_registry(cfg, include_lstm=False)[args.base],
                          feature_subset=schema)
    source = src_factory().fit(X_src, X_src["y"].to_numpy())
    print(f"   {args.base} on {len(X_src)} rows, "
          f"{X_src.index.min().date()} -> {X_src.index.max().date()}")

    rows = []
    print("\n== transfer to each target site ==")
    for country in TARGETS:
        rows += run_site(country, X_src, source, cfg, args.base, schema,
                         X=frames[country])

    if args.also_gbm:
        print("\n== repeat on one site with the boosted model ==")
        gbm_src = partial(model_registry(cfg, include_lstm=False)["gbm_quantile"],
                          feature_subset=schema)().fit(X_src, X_src["y"].to_numpy())
        rows += run_site(TARGETS[0], X_src, gbm_src, cfg, "gbm_quantile", schema,
                         X=frames[TARGETS[0]])

    df = pd.DataFrame(rows)
    save_table(df, "e5_transfer.csv")

    # Fold-averaged view; the per-fold rows stay on disk for the spread.
    keys = ["site", "site_name", "latitude", "strategy", "budget_days", "base"]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
           and c not in ("fold", "latitude", "budget_days")]
    mean = df.groupby(keys, as_index=False)[num].mean()
    mean["mae_sd"] = df.groupby(keys)["mae"].std().to_numpy()
    save_table(mean, "e5_transfer_mean.csv")

    cross = crossover_table(mean[mean["base"] == args.base])
    save_table(cross, "e5_crossover.csv")
    print("\n== crossover: local history needed to beat zero-shot transfer ==")
    print(cross.round(4).to_string(index=False))

    print("\n== figures ==")
    print(" ", plots.plot_transfer_curves(mean[mean["base"] == args.base]).name)
    print(" ", plots.plot_transfer_summary(cross, mean).name)
    print(f"\n== done in {time.time() - t0:.0f}s ==")


if __name__ == "__main__":
    main()
