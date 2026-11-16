"""Retraining policies compared at an equal retraining budget.

The comparison that matters operationally is not "detector A has better
precision than detector B on synthetic change points".  It is: given that you
can afford K retrains a year, does spending them *when a detector fires* beat
spending them *on a calendar*?

Budget-matching is what makes the comparison fair.  An unconstrained
drift-triggered policy that retrains 40 times will beat a quarterly schedule and
prove nothing except that retraining more often helps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import Config
from ..models.base import Forecaster
from ..models.conformal import ConformalizedQuantile
from ..evaluation import metrics as M
from .detectors import ADWINDetector, PageHinkley


@dataclass
class RetrainLog:
    policy: str
    refits: list[pd.Timestamp] = field(default_factory=list)
    triggers: list[pd.Timestamp] = field(default_factory=list)

    @property
    def n_refits(self) -> int:
        return len(self.refits)


def _fit_model(factory: Callable[[], Forecaster], X_hist: pd.DataFrame,
               full: pd.DataFrame) -> Forecaster:
    model = factory()
    y = X_hist["y"].to_numpy()
    try:
        return model.fit(X_hist, y, history=full)
    except TypeError:
        return model.fit(X_hist, y)


def _calibrated(model: Forecaster, X_cal: pd.DataFrame, quantiles) -> object:
    cq = ConformalizedQuantile(model, quantiles, condition_on="horizon")
    try:
        return cq.calibrate(X_cal, X_cal["y"].to_numpy())
    except Exception:  # pragma: no cover - model without quantiles
        return model


def simulate_policy(
    X: pd.DataFrame,
    factory: Callable[[], Forecaster],
    cfg: Config,
    policy: str = "static",
    test_start: pd.Timestamp | None = None,
    detector_name: str = "adwin",
    budget: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, RetrainLog]:
    """Walk forward day by day, refitting according to ``policy``.

    Policies
    --------
    ``static``    fit once at the start of the test period, never refit.
    ``periodic``  refit on a fixed calendar spaced to exactly exhaust the budget.
    ``drift``     refit when the detector fires, subject to a cooldown and the
                  same budget; leftover budget is simply not spent.
    """
    quantiles = np.asarray(cfg.forecast.quantiles, dtype=float)
    budget = cfg.drift.retrain_budget if budget is None else budget
    cal_days = cfg.backtest.calibration_days
    cooldown = pd.Timedelta(days=cfg.drift.cooldown_days)

    if test_start is None:
        test_start = X.index.min() + pd.Timedelta(days=cfg.backtest.min_train_days + cal_days)
    test_start = pd.Timestamp(test_start)
    if test_start.tzinfo is None and X.index.tz is not None:
        test_start = test_start.tz_localize(X.index.tz)

    test_days = pd.DatetimeIndex(sorted({d for d in X.loc[X.index >= test_start].index.normalize()}))
    if len(test_days) == 0:
        raise ValueError("no test days after test_start")

    detector = {"adwin": ADWINDetector(cfg.drift.adwin_delta),
                "page_hinkley": PageHinkley(cfg.drift.ph_delta, cfg.drift.ph_lambda)}[detector_name]
    detector.reset()

    # Calendar of scheduled refits for the periodic policy: budget refits spread
    # evenly across the test period.
    periodic_days: set[pd.Timestamp] = set()
    if policy == "periodic" and budget > 0:
        stride = max(1, len(test_days) // (budget + 1))
        periodic_days = {test_days[min((i + 1) * stride, len(test_days) - 1)]
                         for i in range(budget)}

    log = RetrainLog(policy=policy)
    model = obj = None
    last_refit: pd.Timestamp | None = None
    daily_err_buffer: list[float] = []
    rows = []

    def refit(as_of: pd.Timestamp) -> None:
        nonlocal model, obj, last_refit
        hist = X.loc[X.index < as_of]
        cal_cut = as_of - pd.Timedelta(days=cal_days)
        X_tr, X_cal = hist.loc[hist.index < cal_cut], hist.loc[hist.index >= cal_cut]
        if len(X_tr) < 500 or len(X_cal) < 100:
            X_tr, X_cal = hist, hist.tail(max(200, len(hist) // 5))
        model = _fit_model(factory, X_tr, X)
        obj = _calibrated(model, X_cal, quantiles)
        last_refit = as_of
        log.refits.append(as_of)
        if verbose:
            print(f"    [{policy}] refit at {as_of.date()} on {len(X_tr)} rows")

    refit(test_start)

    for day in test_days:
        day_rows = X.loc[(X.index >= day) & (X.index < day + pd.Timedelta(days=1))]
        if len(day_rows) == 0:
            continue

        point = obj.predict(day_rows)
        try:
            qp = obj.predict_quantiles(day_rows, quantiles)
        except NotImplementedError:
            qp = None
        y = day_rows["y"].to_numpy()

        frame = pd.DataFrame({"y": y, "point": point,
                              "daylight": day_rows["cos_zenith"].to_numpy() > 1e-3,
                              "horizon": day_rows["horizon"].to_numpy()},
                             index=day_rows.index)
        if qp is not None:
            for j, q in enumerate(quantiles):
                frame[f"q{q:.2f}"] = qp[:, j]
        frame["n_refits_so_far"] = log.n_refits
        rows.append(frame)

        # --- observe the outcome, then decide about retraining ---------------
        day_err = float(np.mean(np.abs(y - point)[frame["daylight"].to_numpy()])) \
            if frame["daylight"].any() else np.nan
        daily_err_buffer.append(day_err)

        if policy == "periodic" and day in periodic_days and log.n_refits <= budget:
            refit(day + pd.Timedelta(days=1))
        elif policy == "drift" and np.isfinite(day_err):
            fired = detector.update(day_err)
            if fired:
                log.triggers.append(day)
                detector.reset()
                affordable = log.n_refits <= budget
                cooled = last_refit is None or (day - last_refit) >= cooldown
                if affordable and cooled:
                    refit(day + pd.Timedelta(days=1))

    preds = pd.concat(rows)
    return preds, log


def score_policy(preds: pd.DataFrame, cfg: Config, daylight_only: bool = True) -> dict:
    quantiles = np.asarray(cfg.forecast.quantiles, dtype=float)
    sel = preds["daylight"].to_numpy() if daylight_only else np.ones(len(preds), bool)
    y, p = preds["y"].to_numpy()[sel], preds["point"].to_numpy()[sel]
    qcols = [f"q{q:.2f}" for q in quantiles]
    qp = preds.loc[sel, qcols].to_numpy() if set(qcols) <= set(preds.columns) else None
    return M.evaluate_all(y, p, qp, quantiles, scale=float(np.mean(y) + 1e-9))


def rolling_scores(preds: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Monthly-resolution error trajectory, for the drift figure."""
    sel = preds["daylight"].to_numpy() if "daylight" in preds else np.ones(len(preds), bool)
    d = preds[sel].copy()
    d["abs_err"] = (d["y"] - d["point"]).abs()
    daily = d.groupby(d.index.normalize())["abs_err"].mean()
    return pd.DataFrame({
        "daily_mae": daily,
        "rolling_mae": daily.rolling(window_days, min_periods=max(3, window_days // 3)).mean(),
    })
