"""Shared plumbing: data loading with caching, and the model registry.

Keeping the registry in one place means every experiment scores exactly the
same model definitions -- the forecasting leaderboard, the dispatch study and
the drift study cannot silently drift apart.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config, DATA_PROCESSED
from .data import features as F
from .data.opsd import load_site_series
from .models.base import Forecaster
from .models.baselines import (ClearSkyClimatology, ClimatologicalQuantiles,
                               ProphetBaseline, RidgeForecaster, SeasonalPersistence)
from .models.gbm import GBMPoint, GBMQuantile

DESIGN_CACHE = DATA_PROCESSED / "design_matrix.parquet"


def load_design_matrix(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Build (or reuse) the design matrix for the configured target."""
    cache = DESIGN_CACHE.with_name(f"design_{cfg.forecast.target}_g{cfg.forecast.gate_hour}.parquet")
    if cache.exists() and not force:
        X = pd.read_parquet(cache)
        X.index = pd.DatetimeIndex(X.index)
        return X
    frame = load_site_series(cfg.forecast.target)
    X = F.build_design_matrix(frame, cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(cache)
    return X


def load_raw_frame(cfg: Config) -> pd.DataFrame:
    return load_site_series(cfg.forecast.target)


def load_design_matrix_for(country: str, cfg: Config, normalise: str = "quantile",
                           force: bool = False) -> pd.DataFrame:
    """Design matrix for another bidding zone, with that site's own geometry.

    The clear-sky block is recomputed from the target site's coordinates, which
    is what makes zero-shot transfer meaningful at all: the model never sees a
    raw megawatt or an absolute solar angle, only a normalised output and a
    clear-sky index defined for wherever it is standing.
    """
    import copy

    from .data.opsd import SITES

    if country not in SITES:
        raise KeyError(f"no coordinates registered for {country!r}")
    cache = (DATA_PROCESSED /
             f"design_{country}_{cfg.forecast.target}_g{cfg.forecast.gate_hour}_{normalise}.parquet")
    if cache.exists() and not force:
        X = pd.read_parquet(cache)
        X.index = pd.DatetimeIndex(X.index)
        return X

    site_cfg = copy.deepcopy(cfg)
    site_cfg.site.latitude = SITES[country]["latitude"]
    site_cfg.site.longitude = SITES[country]["longitude"]
    site_cfg.site.name = country
    frame = load_site_series(cfg.forecast.target, country=country, normalise=normalise)
    X = F.build_design_matrix(frame, site_cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(cache)
    return X


def model_registry(cfg: Config, include_lstm: bool = True, fast: bool = False
                   ) -> dict[str, Callable[[], Forecaster]]:
    """Name -> zero-argument factory, so each fold gets a fresh model."""
    qs = tuple(cfg.forecast.quantiles)
    n_est = 150 if fast else 500
    params = dict(n_estimators=n_est)

    reg: dict[str, Callable[[], Forecaster]] = {
        "persistence": SeasonalPersistence,
        "climatology": ClearSkyClimatology,
        "climatology_quantiles": ClimatologicalQuantiles,
        "ridge": RidgeForecaster,
        "gbm_point": partial(GBMPoint, params=params),
        # XGBoost is named in the brief, so it is reported as its own row rather
        # than hidden behind the backend auto-selection, which prefers LightGBM.
        "gbm_point_xgb": partial(GBMPoint, params=params, backend="xgboost"),
        "gbm_quantile": partial(GBMQuantile, quantiles=qs, params=params),
        "gbm_quantile_k": partial(GBMQuantile, quantiles=qs, params=params,
                                  target_space="k"),
    }
    try:  # Prophet is an optional dependency; skip the row if it is absent
        import prophet  # noqa: F401
        reg["prophet"] = ProphetBaseline
    except Exception:  # pragma: no cover
        pass

    if include_lstm:
        try:
            from .models.neural import LSTMQuantile
            reg["lstm_quantile"] = partial(
                LSTMQuantile, quantiles=qs, epochs=8 if fast else 25,
                hidden=64, seed=cfg.seed % 2**31,
            )
        except ImportError:  # pragma: no cover
            pass
    return reg


def quantile_columns(cfg: Config) -> list[str]:
    return [f"q{q:.2f}" for q in cfg.forecast.quantiles]


def save_table(df: pd.DataFrame, name: str, results_dir: Path | None = None) -> Path:
    from .config import RESULTS
    out = (results_dir or RESULTS) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out)
    else:
        df.to_csv(out)
    return out
