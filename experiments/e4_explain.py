"""Experiment 4 - explaining the forecaster.

Answers RQ4.  TreeSHAP attribution on the boosted-tree model, grouped into
feature families so the result is a statement about *what kind* of information
drives the forecast, plus an error-attribution pass that localises failure
modes.

Outputs
-------
results/e4_shap_global.csv, e4_shap_families.csv, e4_error_attribution.csv
results/e4_ablation.csv
figures/10
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from riskaware_pv.config import Config, RESULTS
from riskaware_pv.data.features import feature_columns
from riskaware_pv.evaluation import metrics as M
from riskaware_pv.explain import shap_analysis as S
from riskaware_pv.models.gbm import GBMPoint
from riskaware_pv.pipeline import load_design_matrix, save_table
from riskaware_pv.viz import plots

FAMILY_PREFIXES = {
    "physical (solar geometry)": ("clearsky", "cos_zenith", "solar_", "airmass", "is_daylight"),
    "calendar": ("hod_", "doy_", "hour", "dayofweek", "is_weekend", "horizon"),
    "autoregressive lags": ("y_sameh", "k_sameh", "y_at_origin"),
    "recent state at origin": ("k_roll", "y_roll", "k_recent"),
    "exogenous": ("load_", "price_", "wind_"),
}


def ablation(X: pd.DataFrame, cfg: Config, test_start: pd.Timestamp,
             n_estimators: int = 300) -> pd.DataFrame:
    """Leave-one-family-out: the causal complement to SHAP's correlational view.

    SHAP says which features the fitted model leaned on.  Ablation says which
    families the model could not have done without -- and the two answers differ
    whenever features are redundant, which here they very much are.
    """
    cols = feature_columns(X)
    X_tr = X.loc[X.index < test_start]
    X_te = X.loc[X.index >= test_start]
    X_te = X_te[X_te["cos_zenith"] > 1e-3]
    y_tr, y_te = X_tr["y"].to_numpy(), X_te["y"].to_numpy()

    def fit_score(drop_prefixes=()) -> float:
        keep = [c for c in cols if not any(c.startswith(p) for p in drop_prefixes)]
        # The frame is untouched; only the model's declared input set shrinks,
        # so an ablated family cannot sneak back in as a helper column.
        m = GBMPoint(params=dict(n_estimators=n_estimators), feature_subset=keep)
        m.fit(X_tr, y_tr)
        return M.mae(y_te, m.predict(X_te))

    base = fit_score()
    rows = [{"removed": "nothing (full model)", "mae": base, "delta_mae": 0.0,
             "pct_worse": 0.0}]
    for family, prefixes in FAMILY_PREFIXES.items():
        mae = fit_score(prefixes)
        rows.append({"removed": family, "mae": mae, "delta_mae": mae - base,
                     "pct_worse": 100.0 * (mae - base) / base})
        print(f"   without {family:28s} MAE {mae:.4f}  ({rows[-1]['pct_worse']:+.1f}%)")
    return pd.DataFrame(rows).sort_values("delta_mae", ascending=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load()
    t0 = time.time()
    X = load_design_matrix(cfg)
    test_start = X.index.max() - pd.Timedelta(days=365)
    X_tr, X_te = X.loc[X.index < test_start], X.loc[X.index >= test_start]
    cols = feature_columns(X)

    print("== fitting the model to explain ==")
    model = GBMPoint(params=dict(n_estimators=400))
    model.fit(X_tr, X_tr["y"].to_numpy())
    point = model.predict(X_te)
    day = X_te["cos_zenith"].to_numpy() > 1e-3
    print(f"   test MAE (daylight) = {M.mae(X_te['y'].to_numpy()[day], point[day]):.4f}")

    print("\n== TreeSHAP ==")
    values, Xs = S.tree_shap_values(model._model, X_te, cols, sample=3000, seed=cfg.seed)
    imp = S.global_importance(values, cols)
    save_table(imp, "e4_shap_global.csv")
    print(imp.head(15).round(5).to_string(index=False))

    families = S.grouped_importance(imp)
    save_table(families.to_frame("share"), "e4_shap_families.csv")
    print("\n== attribution by feature family (share of total |SHAP|) ==")
    print((families * 100).round(1).to_string())

    error = pd.Series(X_te["y"].to_numpy() - point, index=X_te.index)
    attrib = S.error_attribution(values, Xs, cols, error)
    save_table(attrib, "e4_error_attribution.csv")
    print("\n== features whose attribution tracks the error ==")
    print(attrib.round(4).to_string(index=False))

    top_feature = imp.iloc[0]["feature"]
    dep = S.dependence_curve(values, Xs, cols, top_feature)
    print("\n== figures ==")
    print(" ", plots.plot_shap(families, imp, dep, top_feature).name)

    if not args.no_ablation:
        print("\n== leave-one-family-out ablation ==")
        abl = ablation(X, cfg, test_start)
        save_table(abl, "e4_ablation.csv")
        print(abl.round(4).to_string(index=False))

    print(f"\n== done in {time.time() - t0:.0f}s ==")


if __name__ == "__main__":
    main()
