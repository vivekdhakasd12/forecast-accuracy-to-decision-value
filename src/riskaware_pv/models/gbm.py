"""Gradient-boosted trees: point and native-quantile heads.

Two backends are supported.  LightGBM is preferred (fast, native pinball
objective); scikit-learn's ``HistGradientBoostingRegressor`` is a drop-in
fallback with the same quantile loss and no OpenMP dependency, so the study
reproduces on a machine without ``libomp``.

Quantile crossing (the 0.9 quantile predicted below the 0.75) is a real and
common failure of independently fitted quantile models.  It is repaired here by
*rearrangement* -- sorting the predicted quantiles per row -- which Chernozhukov
et al. (2010) show never increases the estimation error.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data.features import feature_columns
from .base import Forecaster, ResidualQuantileMixin

try:  # pragma: no cover - environment dependent
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False

try:  # pragma: no cover
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingRegressor

DEFAULT_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)


# Any |value| above this is treated as missing.  Real inputs here top out
# around 8e4 (national load in MW), so the bound only ever catches a numerical
# blow-up, and catching it is what keeps a backend swap from failing outright.
FEATURE_ABS_MAX = 1e12


def _prep(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = X[cols].to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return np.where(np.abs(arr) > FEATURE_ABS_MAX, np.nan, arr)


class GBMPoint(Forecaster, ResidualQuantileMixin):
    """Squared-error boosted trees; quantiles via conditional residuals."""

    name = "gbm_point"

    def __init__(self, params: dict | None = None, backend: str = "auto", **kw):
        super().__init__(**kw)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.backend = self._resolve_backend(backend)

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend != "auto":
            return backend
        if _HAS_LGB:
            return "lightgbm"
        if _HAS_XGB:
            return "xgboost"
        return "sklearn"

    def _make(self, objective: str | None = None, alpha: float | None = None):
        p = dict(self.params)
        if self.backend == "lightgbm":
            kw = dict(
                n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                subsample=p["subsample"], subsample_freq=p["subsample_freq"],
                colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
                verbose=-1, n_jobs=-1,
            )
            if objective == "quantile":
                kw.update(objective="quantile", alpha=alpha)
            return lgb.LGBMRegressor(**kw)
        if self.backend == "xgboost" and objective != "quantile":
            return xgb.XGBRegressor(
                n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                max_depth=6, min_child_weight=p["min_child_samples"] / 10,
                subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                reg_lambda=p["reg_lambda"], n_jobs=-1, tree_method="hist",
            )
        kw = dict(
            max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
            max_leaf_nodes=p["num_leaves"], min_samples_leaf=p["min_child_samples"],
            l2_regularization=p["reg_lambda"], early_stopping=False,
        )
        if objective == "quantile":
            kw.update(loss="quantile", quantile=alpha)
        return HistGradientBoostingRegressor(**kw)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMPoint":
        self._cols = self._features(X)
        rows = self._train_rows(X)
        Xf = _prep(X.loc[rows], self._cols)
        z = self._to_learning_target(X, y)[rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = self._make().fit(np.nan_to_num(Xf, nan=-999.0), z)
        self._fit_residuals(X, y, self.predict(X))
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = np.nan_to_num(_prep(X, self._cols), nan=-999.0)
        return self._from_learning_target(X, self._model.predict(Xf))

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        q = self._residual_quantiles(X, self.predict(X), quantiles)
        return self._postprocess(X, np.sort(q, axis=1))

    def feature_importance(self) -> pd.Series:
        model = self._model
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:  # pragma: no cover - sklearn hist GBM has no native importance
            imp = np.zeros(len(self._cols))
        return pd.Series(imp, index=self._cols).sort_values(ascending=False)


class GBMQuantile(Forecaster):
    """One boosted model per quantile, trained on the pinball loss.

    This is the workhorse probabilistic forecaster of the study: it makes no
    Gaussian assumption, which matters enormously for PV where the predictive
    distribution is bounded at zero, bounded above by the clear-sky ceiling, and
    strongly skewed in between.
    """

    name = "gbm_quantile"
    supports_native_quantiles = True

    def __init__(self, quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
                 params: dict | None = None, backend: str = "auto", **kw):
        super().__init__(**kw)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._helper = GBMPoint(params=self.params, backend=backend,
                                target_space=self.target_space,
                                feature_subset=self.feature_subset)
        self.backend = self._helper.backend

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMQuantile":
        self._cols = self._features(X)
        rows = self._train_rows(X)
        Xf = np.nan_to_num(_prep(X.loc[rows], self._cols), nan=-999.0)
        z = self._to_learning_target(X, y)[rows]
        self._models = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for q in self.quantiles:
                self._models[q] = self._helper._make("quantile", q).fit(Xf, z)
        self._fitted = True
        return self

    def _raw_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        Xf = np.nan_to_num(_prep(X, self._cols), nan=-999.0)
        available = np.array(self.quantiles)
        preds = np.column_stack([self._models[q].predict(Xf) for q in self.quantiles])
        preds = np.sort(preds, axis=1)  # rearrangement: removes quantile crossing
        want = np.asarray(quantiles, dtype=float)
        if len(want) == len(available) and np.allclose(want, available):
            return preds
        # Linear interpolation across the fitted quantile grid for any other level.
        out = np.empty((len(X), len(want)))
        for i in range(len(X)):
            out[i] = np.interp(want, available, preds[i])
        return out

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> np.ndarray:
        quantiles = self.quantiles if quantiles is None else quantiles
        raw = self._raw_quantiles(X, quantiles)
        return self._from_learning_target(X, raw)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecast = predictive median (the pinball-optimal point)."""
        med = self._raw_quantiles(X, [0.5])[:, 0]
        return self._from_learning_target(X, med)

    def feature_importance(self, q: float = 0.5) -> pd.Series:
        model = self._models[min(self._models, key=lambda x: abs(x - q))]
        imp = getattr(model, "feature_importances_", np.zeros(len(self._cols)))
        return pd.Series(imp, index=self._cols).sort_values(ascending=False)
