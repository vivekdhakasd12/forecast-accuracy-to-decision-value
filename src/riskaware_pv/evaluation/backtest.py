"""Rolling-origin backtesting with an explicit conformal calibration slice.

Every fold has three disjoint, chronologically ordered blocks:

    [------------ train ------------][-- calibration --][-- test --]

The calibration block exists so that split-conformal calibration never sees the
data it is scored on.  Folds advance by ``step_days`` and never shuffle: with a
seasonal series, k-fold cross-validation would train on July to predict June and
report skill that no operational system can have.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..models.base import Forecaster
from ..models.conformal import ConformalizedQuantile
from . import metrics as M


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[pd.Timestamp, pd.Timestamp]
    calib: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]

    def slice(self, X: pd.DataFrame, part: str) -> pd.DataFrame:
        lo, hi = getattr(self, part)
        return X.loc[(X.index >= lo) & (X.index < hi)]

    def __str__(self) -> str:  # pragma: no cover - logging only
        return (f"fold{self.index}: train {self.train[0].date()}->{self.train[1].date()} | "
                f"cal ->{self.calib[1].date()} | test ->{self.test[1].date()}")


def make_folds(index: pd.DatetimeIndex, cfg: Config, n_folds: int | None = None
               ) -> list[Fold]:
    """Expanding-window folds covering the tail of the sample."""
    bt = cfg.backtest
    start, end = index.min(), index.max()
    test_len = pd.Timedelta(days=bt.test_days)
    cal_len = pd.Timedelta(days=bt.calibration_days)
    min_train = pd.Timedelta(days=bt.min_train_days)

    first_test_start = start + min_train + cal_len
    if first_test_start >= end:
        raise ValueError("sample too short for the configured backtest windows")

    folds: list[Fold] = []
    test_start = first_test_start
    i = 0
    while test_start < end:
        test_end = min(test_start + test_len, end)
        if test_end - test_start < pd.Timedelta(days=30):
            break
        folds.append(Fold(
            index=i,
            train=(start, test_start - cal_len),
            calib=(test_start - cal_len, test_start),
            test=(test_start, test_end),
        ))
        i += 1
        test_start = test_end
    if n_folds is not None:
        folds = folds[-n_folds:] if n_folds > 0 else folds
    return folds


def _can_quantile(model: Forecaster, probe: pd.DataFrame, quantiles) -> bool:
    """Does this fitted model actually produce a quantile matrix?"""
    try:
        out = model.predict_quantiles(probe, quantiles)
        return out is not None and out.shape[1] == len(quantiles)
    except (NotImplementedError, AttributeError, KeyError):
        return False


def _fit(model_factory: Callable[[], Forecaster], X_tr: pd.DataFrame,
         history: pd.DataFrame | None = None) -> Forecaster:
    model = model_factory()
    y_tr = X_tr["y"].to_numpy()
    try:  # the LSTM needs the raw history to build its encoder sequences
        return model.fit(X_tr, y_tr, history=history)
    except TypeError:
        return model.fit(X_tr, y_tr)


def run_backtest(
    X: pd.DataFrame,
    model_factories: dict[str, Callable[[], Forecaster]],
    cfg: Config,
    folds: Sequence[Fold] | None = None,
    conformalise: bool = True,
    store_predictions: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score every model on every fold.

    Returns ``(scores, predictions)`` where ``scores`` has one row per
    (fold, model, variant) and ``predictions`` holds the per-timestamp point and
    quantile forecasts needed by the dispatch and drift stages.
    """
    quantiles = np.asarray(cfg.forecast.quantiles, dtype=float)
    folds = folds if folds is not None else make_folds(X.index, cfg)
    rows, preds = [], []

    for fold in folds:
        X_tr, X_cal, X_te = (fold.slice(X, p) for p in ("train", "calib", "test"))
        if len(X_tr) < 500 or len(X_te) < 100:
            continue
        if verbose:
            print(f"  {fold}  (n_train={len(X_tr)}, n_cal={len(X_cal)}, n_test={len(X_te)})")

        y_te = X_te["y"].to_numpy()
        scale = float(np.nanmean(X_tr["y"].to_numpy()) + 1e-9)
        ref_mae = M.mae(y_te, np.nan_to_num(X_te["y_sameh_d1"].to_numpy()))

        for name, factory in model_factories.items():
            fit_on = pd.concat([X_tr, X_cal]) if not conformalise else X_tr
            model = _fit(factory, fit_on, history=X)

            variants: list[tuple[str, object]] = [("raw", model)]
            if conformalise and _can_quantile(model, X_cal.head(4), quantiles):
                # Any model that emits quantiles can be conformalised -- not just
                # the natively probabilistic ones -- so the comparison between
                # "raw" and "calibrated" is available for the whole zoo.
                cq = ConformalizedQuantile(model, quantiles, condition_on="horizon")
                cq.calibrate(X_cal, X_cal["y"].to_numpy())
                variants.append(("cqr", cq))

            day = X_te["cos_zenith"].to_numpy() > 1e-3
            for variant, obj in variants:
                point = obj.predict(X_te)
                try:
                    qp = obj.predict_quantiles(X_te, quantiles)
                except NotImplementedError:
                    qp = None

                # Night hours are trivially correct for every model and dilute
                # every average, so daylight-only scores are reported alongside.
                for subset, sel in (("all", np.ones(len(X_te), bool)), ("daylight", day)):
                    if sel.sum() < 50:
                        continue
                    row = {"fold": fold.index, "model": name, "variant": variant,
                           "subset": subset, "test_start": fold.test[0],
                           "test_end": fold.test[1], "n_test": int(sel.sum())}
                    row.update(M.evaluate_all(
                        y_te[sel], point[sel], qp[sel] if qp is not None else None,
                        quantiles, float(np.mean(y_te[sel]) + 1e-9),
                        M.mae(y_te[sel], np.nan_to_num(X_te["y_sameh_d1"].to_numpy()[sel])),
                    ))
                    rows.append(row)

                if store_predictions:
                    frame = pd.DataFrame(
                        {"y": y_te, "point": point, "horizon": X_te["horizon"].to_numpy(),
                         "daylight": day, "fold": fold.index, "model": name,
                         "variant": variant},
                        index=X_te.index,
                    )
                    if qp is not None:
                        for j, q in enumerate(quantiles):
                            frame[f"q{q:.2f}"] = qp[:, j]
                    preds.append(frame)

    scores = pd.DataFrame(rows)
    predictions = pd.concat(preds) if preds else pd.DataFrame()
    return scores, predictions


def summarise(scores: pd.DataFrame, by: Iterable[str] = ("model", "variant"),
              subset: str = "daylight") -> pd.DataFrame:
    """Fold-averaged leaderboard with a paired standard error across folds."""
    if "subset" in scores.columns and subset is not None:
        scores = scores[scores["subset"] == subset]
    keep = [c for c in scores.columns
            if c not in {"fold", "test_start", "test_end", "n_test", "subset", *by}
            and pd.api.types.is_numeric_dtype(scores[c])]
    by = list(by)
    grouped = scores.groupby(by)
    out = grouped[keep].mean()
    n_folds = grouped.size()
    for col in ("mae", "crps", "pinball"):
        if col in keep:
            # Fold-to-fold standard error: with only a handful of folds this is
            # the honest uncertainty on the leaderboard, and it is usually large
            # enough to make neighbouring rows statistically indistinguishable.
            out[f"{col}_se"] = grouped[col].std() / np.sqrt(n_folds.clip(lower=1))
    return out.sort_values("crps" if "crps" in out.columns else "mae")
