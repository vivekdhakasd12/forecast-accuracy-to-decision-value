"""Small-data strategy: transferring a forecaster to a site with little history.

The realistic version of "small data" in this domain is not a small dataset in
the abstract. It is a *new site*: an operator commissions an installation, has a
fortnight of generation history, and must nonetheless submit a day-ahead
schedule tomorrow. The question is whether a model trained somewhere else, on
years of data, is worth more than a model trained here on two weeks.

Four strategies span the answer:

``local``      fit only on the local history available. The honest floor.
``zero_shot``  apply the source model unchanged, using *no* local data at all.
               Possible only because the target is normalised and the physical
               features are recomputed for the target site's own coordinates, so
               the model never sees a raw megawatt or an absolute solar angle.
``finetune``   source point forecast plus a correction learned locally. The
               correction is a plain linear model on the same features, so it can
               absorb a systematic site offset without needing enough data to
               relearn the whole mapping.
``pooled``     fit on source and local data together with the local rows
               upweighted, letting the learner decide how much to borrow.

The quantity of interest is the crossover: how many days of local history are
needed before training locally beats borrowing from a different country.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data.features import feature_columns
from .models.base import Forecaster, ResidualQuantileMixin

STRATEGIES = ("local", "zero_shot", "finetune", "pooled")


@dataclass
class TransferResult:
    site: str
    strategy: str
    budget_days: int
    n_train: int
    mae: float
    rmse: float
    crps: float
    picp90: float


class TransferForecaster(Forecaster, ResidualQuantileMixin):
    """Wraps a source-trained model with one of the transfer strategies."""

    name = "transfer"

    def __init__(self, strategy: str, source: Forecaster | None = None,
                 local_factory: Callable[[], Forecaster] | None = None,
                 source_frame: pd.DataFrame | None = None,
                 local_weight: float = 5.0, ridge_alpha: float = 10.0, **kw):
        super().__init__(**kw)
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.source = source
        self.local_factory = local_factory
        self.source_frame = source_frame
        self.local_weight = local_weight
        self.ridge_alpha = ridge_alpha

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TransferForecaster":
        self._cols = self._features(X)

        if self.strategy == "zero_shot":
            # No local data is consumed, by definition. Predictive spread comes
            # from the source model, which is the only honest option when the
            # premise is that nothing local has been observed yet.
            self._inner = self.source

        elif self.strategy == "local":
            self._inner = self.local_factory().fit(X, y)

        elif self.strategy == "finetune":
            self._inner = self.source
            base = self.source.predict(X)
            self._correction = make_pipeline(
                StandardScaler(), Ridge(alpha=self.ridge_alpha)
            ).fit(self._design(X), y - base)

        elif self.strategy == "pooled":
            if self.source_frame is None:
                raise ValueError("pooled strategy needs the source frame")
            pooled = pd.concat([self.source_frame, X])
            y_pooled = np.concatenate([self.source_frame["y"].to_numpy(), y])
            weights = np.concatenate([
                np.ones(len(self.source_frame)),
                np.full(len(X), self.local_weight),
            ])
            self._inner = self._fit_weighted(pooled, y_pooled, weights)

        if self.strategy != "zero_shot":
            self._fit_residuals(X, y, self._point(X))
        self._fitted = True
        return self

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self._cols].to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        med = getattr(self, "_median", None)
        if med is None:
            with np.errstate(all="ignore"):
                med = np.nan_to_num(np.nanmedian(arr, axis=0), nan=0.0)
            self._median = med
        return np.clip(np.where(np.isfinite(arr), arr, med), -1e6, 1e6)

    def _fit_weighted(self, X: pd.DataFrame, y: np.ndarray,
                      weights: np.ndarray) -> Forecaster:
        """Upweighting local rows is what makes 'pooled' different from 'source'.

        Sample weights are applied by resampling rather than passed through,
        because the shared model interface has no weight argument and adding one
        for a single strategy would complicate every other model.
        """
        rng = np.random.default_rng(0)
        p = weights / weights.sum()
        n = min(len(X), 60_000)
        take = rng.choice(len(X), size=n, replace=True, p=p)
        take.sort()
        return self.local_factory().fit(X.iloc[take], y[take])

    # -------------------------------------------------------------- predict
    def _point(self, X: pd.DataFrame) -> np.ndarray:
        out = self._inner.predict(X)
        if self.strategy == "finetune":
            out = out + self._correction.predict(self._design(X))
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._postprocess(X, self._point(X))

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        if self.strategy == "zero_shot":
            return self.source.predict_quantiles(X, quantiles)
        q = self._residual_quantiles(X, self._point(X), quantiles)
        return self._postprocess(X, np.sort(q, axis=1))


def local_slice(X: pd.DataFrame, test_start: pd.Timestamp, budget_days: int
                ) -> pd.DataFrame:
    """The most recent ``budget_days`` of local history before the test period.

    Recency matters: a new site's operator has the last fortnight, not a
    fortnight sampled from two years ago.
    """
    lo = test_start - pd.Timedelta(days=budget_days)
    return X.loc[(X.index >= lo) & (X.index < test_start)]


def stable_crossover(budgets, local_mae, zero_shot_mae: float) -> float:
    """Smallest budget from which local-only *stays* at or below zero-shot.

    Defined once and shared by the results table and the figure, because the two
    disagreeing about which number is the crossover is exactly the kind of quiet
    inconsistency a reader cannot check.
    """
    budgets = np.asarray(budgets)
    order = np.argsort(budgets)
    budgets, local_mae = budgets[order], np.asarray(local_mae)[order]
    wins = local_mae <= zero_shot_mae
    stable = [budgets[i] for i in range(len(budgets)) if wins[i:].all()]
    return float(min(stable)) if stable else float("nan")
