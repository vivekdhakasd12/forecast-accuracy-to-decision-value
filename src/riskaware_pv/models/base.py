"""Common interface for every forecaster in the study.

All models expose the same three-method contract so the backtester, the
conformal wrapper and the dispatch layer never need to know which model they
hold:

    fit(X, y) -> self
    predict(X) -> (n,)              point forecast, original units
    predict_quantiles(X, qs) -> (n, len(qs))

Two cross-cutting concerns are handled here rather than in each model:

* **Target space.**  A PV series can be learned directly (``y``) or as a
  clear-sky index (``k = y / clearsky``).  The second removes the deterministic
  diurnal/seasonal cycle before learning; whether it actually helps is an
  ablation we run rather than an assumption we make.
* **Night clamping.**  Predicting a non-zero value when the sun is below the
  horizon is not a modelling question, it is a bug.  Every prediction is
  clamped to zero there, for every model, so baselines are not unfairly
  penalised relative to the learners.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

Y_MAX = 1.2  # capacity factors above this are non-physical after QC


class Forecaster(ABC):
    """Base class providing target-space handling and night clamping."""

    name: str = "forecaster"
    supports_native_quantiles: bool = False

    def __init__(self, target_space: str = "y", clip: tuple[float, float] = (0.0, Y_MAX),
                 feature_subset: list[str] | None = None):
        if target_space not in {"y", "k"}:
            raise ValueError("target_space must be 'y' or 'k'")
        self.target_space = target_space
        self.clip = clip
        # An explicit subset lets the ablation study control the input set
        # exactly, instead of relying on which helper columns happen to be
        # present in the frame -- which would silently re-admit an ablated
        # feature through the back door.
        self.feature_subset = list(feature_subset) if feature_subset else None
        self._fitted = False

    def _features(self, X: pd.DataFrame) -> list[str]:
        from ..data.features import feature_columns
        if self.feature_subset is not None:
            return [c for c in self.feature_subset if c in X.columns]
        return feature_columns(X)

    # ---------------- target-space plumbing ---------------------------------
    def _to_learning_target(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        if self.target_space == "y":
            return y
        cs = np.clip(X["clearsky_norm"].to_numpy(), 1e-3, None)
        return np.clip(y / cs, 0.0, 1.5)

    def _from_learning_target(self, X: pd.DataFrame, z: np.ndarray) -> np.ndarray:
        if self.target_space == "y":
            out = z
        else:
            cs = np.clip(X["clearsky_norm"].to_numpy(), 0.0, None)
            out = z * cs[:, None] if z.ndim == 2 else z * cs
        return self._postprocess(X, out)

    def _postprocess(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        pred = np.clip(pred, *self.clip)
        night = X["cos_zenith"].to_numpy() <= 1e-3
        if pred.ndim == 2:
            pred[night, :] = 0.0
        else:
            pred[night] = 0.0
        return pred

    def _train_rows(self, X: pd.DataFrame) -> np.ndarray:
        """Rows the model actually learns from.

        In clear-sky-index space, night rows carry no information (k is
        undefined at zero irradiance) and would dominate the loss with trivial
        zeros, so they are excluded from fitting -- never from evaluation.
        """
        if self.target_space == "k":
            return X["cos_zenith"].to_numpy() > 1e-3
        return np.ones(len(X), dtype=bool)

    # ---------------- contract ----------------------------------------------
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Forecaster": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        """Default: shift a point forecast by residual quantiles (see mixin)."""
        raise NotImplementedError(f"{self.name} has no quantile head")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r}, space={self.target_space!r})"


class ResidualQuantileMixin:
    """Turn any point forecaster into a probabilistic one.

    Residuals are pooled *per horizon and per clear-sky regime* rather than
    globally: a 35-hour-ahead forecast on a broken-cloud day is far less certain
    than a 13-hour-ahead one on a clear day, and a single global residual
    distribution would smear those two cases into one uninformative interval.
    """

    _resid_groups: dict
    _resid_global: np.ndarray

    def _fit_residuals(self, X: pd.DataFrame, y: np.ndarray, point: np.ndarray,
                       n_regimes: int = 3) -> None:
        resid = y - point
        horizon = (X["horizon"].to_numpy() if "horizon" in X
                   else np.zeros(len(X)))
        regime = self._regime(X, n_regimes)
        self._n_regimes = n_regimes
        self._resid_global = resid
        self._resid_groups = {}
        for h in np.unique(horizon):
            for r in range(n_regimes):
                sel = (horizon == h) & (regime == r)
                if sel.sum() >= 30:
                    self._resid_groups[(h, r)] = resid[sel]

    @staticmethod
    def _regime(X: pd.DataFrame, n_regimes: int) -> np.ndarray:
        """Coarse cloudiness state at the forecast origin (0=clear .. 2=overcast)."""
        for col in ("k_roll24_mean", "k_recent6", "k_roll72_mean"):
            if col in X.columns:
                k = np.nan_to_num(X[col].to_numpy(), nan=0.5)
                break
        else:  # no cloud-state column survived an ablation -> one global pool
            k = np.full(len(X), 0.5)
        edges = np.linspace(0.0, 1.0, n_regimes + 1)[1:-1]
        return np.digitize(k, edges)

    def _residual_quantiles(self, X: pd.DataFrame, point: np.ndarray,
                            quantiles) -> np.ndarray:
        horizon = (X["horizon"].to_numpy() if "horizon" in X
                   else np.zeros(len(X)))
        regime = self._regime(X, getattr(self, "_n_regimes", 3))
        qs = np.asarray(quantiles, dtype=float)
        out = np.empty((len(X), len(qs)))
        for i in range(len(X)):
            pool = self._resid_groups.get((horizon[i], regime[i]))
            if pool is None or len(pool) < 30:
                pool = self._resid_global
            out[i] = point[i] + np.quantile(pool, qs)
        return out
