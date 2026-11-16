"""SHAP attribution for the tree forecaster.

Two questions are asked, and they are different:

1. *Which features carry the signal?*  Global mean |SHAP|.  This is the honest
   version of "feature importance": gain-based importances from boosted trees
   are biased towards high-cardinality features and are not comparable across
   models.
2. *Which features drive the errors?*  Correlating per-sample SHAP values with
   the realised error localises failure modes -- e.g. if the clear-sky feature's
   attribution correlates with error on overcast days, the physical prior is
   being trusted in exactly the regime where it does not hold.

Only TreeSHAP is used (exact and fast for GBMs); the sample is subsetted because
the interaction-free values converge quickly and the full matrix is not needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import shap
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False


def tree_shap_values(model, X: pd.DataFrame, columns: list[str],
                     sample: int = 3000, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """Exact TreeSHAP values on a random subsample of rows."""
    if not _HAS_SHAP:
        raise ImportError("shap is required for this analysis")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    idx.sort()
    Xs = X.iloc[idx]
    arr = np.nan_to_num(Xs[columns].to_numpy(dtype=float), nan=-999.0)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(arr, check_additivity=False)
    if isinstance(values, list):  # pragma: no cover - multi-output guard
        values = values[0]
    return np.asarray(values), Xs


def global_importance(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature, plus the signed mean (direction of influence)."""
    return (
        pd.DataFrame({
            "feature": columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "std_shap": values.std(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def grouped_importance(imp: pd.DataFrame) -> pd.Series:
    """Collapse features into interpretable families.

    Reviewers care whether the model runs on physics, on autoregression, or on
    the exogenous series -- not whether ``doy_sin2`` outranks ``doy_cos2``.
    """
    def family(name: str) -> str:
        if name.startswith(("clearsky", "cos_zenith", "solar_", "airmass", "is_daylight")):
            return "physical (solar geometry)"
        if name.startswith(("hod_", "doy_", "hour", "dayofweek", "is_weekend", "horizon")):
            return "calendar"
        if name.startswith(("y_sameh", "k_sameh", "y_at_origin")):
            return "autoregressive lags"
        if name.startswith(("k_roll", "y_roll", "k_recent")):
            return "recent state at origin"
        if name.startswith(("load_", "price_", "wind_")):
            return "exogenous"
        return "other"

    out = imp.assign(family=imp["feature"].map(family))
    return (out.groupby("family")["mean_abs_shap"].sum()
            .sort_values(ascending=False)
            .pipe(lambda s: s / s.sum()))


def error_attribution(values: np.ndarray, Xs: pd.DataFrame, columns: list[str],
                      error: pd.Series, top_k: int = 12) -> pd.DataFrame:
    """Correlation between each feature's attribution and the realised error."""
    err = error.reindex(Xs.index).to_numpy()
    ok = np.isfinite(err)
    rows = []
    for j, col in enumerate(columns):
        v = values[ok, j]
        if v.std() < 1e-12:
            continue
        rows.append({"feature": col,
                     "corr_with_error": float(np.corrcoef(v, err[ok])[0, 1]),
                     "mean_abs_shap": float(np.abs(values[:, j]).mean())})
    out = pd.DataFrame(rows).dropna()
    out["abs_corr"] = out["corr_with_error"].abs()
    return out.sort_values("abs_corr", ascending=False).head(top_k).reset_index(drop=True)


def dependence_curve(values: np.ndarray, Xs: pd.DataFrame, columns: list[str],
                     feature: str, bins: int = 20) -> pd.DataFrame:
    """Binned SHAP dependence: how attribution varies across a feature's range."""
    j = columns.index(feature)
    x = Xs[feature].to_numpy(dtype=float)
    v = values[:, j]
    ok = np.isfinite(x)
    x, v = x[ok], v[ok]
    if len(np.unique(x)) < 3:
        return pd.DataFrame(columns=["bin_center", "mean_shap", "n"])
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    which = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = which == b
        if sel.sum() >= 5:
            rows.append({"bin_center": float(np.mean(x[sel])),
                         "mean_shap": float(np.mean(v[sel])), "n": int(sel.sum())})
    return pd.DataFrame(rows)
