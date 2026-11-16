"""Reference models.

A forecasting paper lives or dies by its baselines.  Solar generation is so
strongly driven by astronomy that a naive method can look excellent on RMSE;
if the learned models are not compared against *properly implemented* naive
methods -- including their probabilistic versions -- any reported skill is
uninterpretable.  All four baselines below therefore emit calibratable
predictive distributions, not just point values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.features import feature_columns
from .base import Forecaster, ResidualQuantileMixin


class SeasonalPersistence(Forecaster, ResidualQuantileMixin):
    """Yesterday-at-this-hour, using the most recent *legal* same-hour value.

    This is the operational default at many small sites and is a genuinely hard
    baseline at 24 h lead time.
    """

    name = "persistence"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SeasonalPersistence":
        point = self._raw_point(X)
        self._fit_residuals(X, y, point)
        self._fitted = True
        return self

    @staticmethod
    def _raw_point(X: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(X["y_sameh_d1"].to_numpy(), nan=0.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._postprocess(X, self._raw_point(X).copy())

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        q = self._residual_quantiles(X, self._raw_point(X), quantiles)
        return self._postprocess(X, np.sort(q, axis=1))


class ClearSkyClimatology(Forecaster, ResidualQuantileMixin):
    """Physical clear-sky profile scaled by the historical clear-sky index.

    ``E[y | hour, month] = clearsky(hour) * mean_k(hour, month)``.  Encodes the
    seasonal cycle without any learning, and is the fair reference for asking
    "did the model learn anything beyond climate?".
    """

    name = "climatology"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ClearSkyClimatology":
        cs = np.clip(X["clearsky_norm"].to_numpy(), 1e-3, None)
        k = np.clip(y / cs, 0.0, 1.5)
        key = pd.MultiIndex.from_arrays(
            [X.index.month, X.index.hour], names=["month", "hour"]
        )
        self._k_table = pd.Series(k, index=key).groupby(level=[0, 1]).median()
        self._k_default = float(np.median(k))
        self._fit_residuals(X, y, self._raw_point(X))
        self._fitted = True
        return self

    def _raw_point(self, X: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([X.index.month, X.index.hour])
        k = self._k_table.reindex(key).to_numpy()
        k = np.nan_to_num(k, nan=self._k_default)
        return k * X["clearsky_norm"].to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._postprocess(X, self._raw_point(X))

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        q = self._residual_quantiles(X, self._raw_point(X), quantiles)
        return self._postprocess(X, np.sort(q, axis=1))


class RidgeForecaster(Forecaster, ResidualQuantileMixin):
    """Regularised linear model on the full feature block.

    Separates "the features are informative" from "the model is expressive":
    if Ridge closes most of the gap to the GBM, the story is about feature
    engineering, not about gradient boosting.
    """

    name = "ridge"

    def __init__(self, alpha: float | str = 3.0, **kw):
        super().__init__(**kw)
        # ``alpha="cv"`` selects the penalty by generalised cross-validation.
        # This matters when the model is fitted on a fortnight of data: a fixed
        # penalty tuned for three years of history leaves a 55-feature ridge
        # badly under-regularised on 336 rows, and the resulting instability
        # would make the local-only baseline a strawman in the transfer study.
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RidgeForecaster":
        self._cols = self._features(X)
        rows = self._train_rows(X)
        Xf = X.loc[rows, self._cols].to_numpy(dtype=float)
        z = self._to_learning_target(X, y)[rows]
        with np.errstate(all='ignore'):
            med = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan), axis=0)
        self._imputer = np.nan_to_num(med, nan=0.0)
        Xf = self._impute(Xf)
        if self.alpha == "cv":
            from sklearn.linear_model import RidgeCV
            est = RidgeCV(alphas=np.logspace(-1, 4, 24))
        else:
            est = Ridge(alpha=float(self.alpha))
        self._model = make_pipeline(StandardScaler(), est).fit(Xf, z)
        self._fit_residuals(X, y, self.predict(X))
        self._fitted = True
        return self

    def _impute(self, arr: np.ndarray) -> np.ndarray:
        out = np.where(np.isfinite(arr), arr, self._imputer)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = self._impute(X[self._cols].to_numpy(dtype=float))
        return self._from_learning_target(X, self._model.predict(Xf))

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        q = self._residual_quantiles(X, self.predict(X), quantiles)
        return self._postprocess(X, np.sort(q, axis=1))


class ClimatologicalQuantiles(Forecaster):
    """Unconditional predictive distribution per (hour, month).

    The weakest possible probabilistic forecast that is still calibrated.  It
    anchors the CRPS scale: any model that cannot beat it has learned nothing
    about *this* day, only about this time of year.
    """

    name = "climatology_quantiles"
    supports_native_quantiles = True

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ClimatologicalQuantiles":
        self._table: dict[tuple[int, int], np.ndarray] = {}
        key_month = X.index.month.to_numpy()
        key_hour = X.index.hour.to_numpy()
        for m in range(1, 13):
            for h in range(24):
                sel = (key_month == m) & (key_hour == h)
                if sel.sum() >= 10:
                    self._table[(m, h)] = y[sel]
        self._pool = y
        self._fitted = True
        return self

    def _sample_pool(self, X: pd.DataFrame):
        for m, h in zip(X.index.month, X.index.hour):
            yield self._table.get((m, h), self._pool)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.array([np.median(p) for p in self._sample_pool(X)])
        return self._postprocess(X, out)

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        qs = np.asarray(quantiles, dtype=float)
        out = np.array([np.quantile(p, qs) for p in self._sample_pool(X)])
        return self._postprocess(X, out)


class ProphetBaseline(Forecaster, ResidualQuantileMixin):
    """Prophet: additive decomposition into trend and Fourier seasonality.

    Included because the brief names Prophet-style baselines, and because it is
    the reference point most practitioners reach for first on a seasonal series.
    It is a *univariate* method: it sees the timestamp and the history, and none
    of the engineered features, so it cannot use the recent cloud state that
    Section 5.4 identifies as the only genuinely irreplaceable signal. At a
    day-ahead horizon its forecast is therefore its seasonal decomposition, which
    makes it a close relative of the clear-sky climatology baseline -- a useful
    thing to be able to show rather than assert.

    Two configuration choices are deliberate. The daily Fourier order is raised
    well above Prophet's default of 4, because a PV day is flat at zero all night
    and then rises and falls sharply, which a low-order series cannot follow.
    The changepoint prior is tightened because the target is already normalised,
    so a flexible trend has nothing real to fit and would extrapolate noise
    across a twelve-month test period.
    """

    name = "prophet"

    def __init__(self, daily_order: int = 10, yearly_order: int = 6,
                 changepoint_prior_scale: float = 0.01, **kw):
        super().__init__(**kw)
        self.daily_order = daily_order
        self.yearly_order = yearly_order
        self.changepoint_prior_scale = changepoint_prior_scale

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ProphetBaseline":
        import logging

        from prophet import Prophet

        for noisy in ("prophet", "cmdstanpy"):
            logging.getLogger(noisy).setLevel(logging.CRITICAL)

        idx = pd.DatetimeIndex(X.index)
        df = pd.DataFrame({"ds": idx.tz_localize(None), "y": y})
        self._model = Prophet(
            growth="flat",
            daily_seasonality=self.daily_order,
            yearly_seasonality=self.yearly_order,
            weekly_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            uncertainty_samples=0,   # intervals come from the residual mixin
        )
        self._model.fit(df)
        self._fit_residuals(X, y, self._raw_point(X))
        self._fitted = True
        return self

    def _raw_point(self, X: pd.DataFrame) -> np.ndarray:
        idx = pd.DatetimeIndex(X.index).tz_localize(None)
        out = self._model.predict(pd.DataFrame({"ds": idx}))["yhat"].to_numpy()
        return np.nan_to_num(out, nan=0.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._postprocess(X, self._raw_point(X))

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        q = self._residual_quantiles(X, self._raw_point(X), quantiles)
        return self._postprocess(X, np.sort(q, axis=1))
