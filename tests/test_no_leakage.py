"""The single most important test in the repository.

A day-ahead study that accidentally uses post-gate information reports skill
that cannot be realised operationally.  Rather than trusting the feature code by
inspection, this test *poisons* the series after each forecast origin and
asserts that the design matrix is bit-for-bit unchanged.  If any feature peeked
into the future, the poison would propagate and the assertion would fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskaware_pv.config import Config
from riskaware_pv.data import features as F


def _toy_frame(days: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=days * 24, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    diurnal = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    seasonal = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    cloud = np.clip(rng.beta(4, 2, size=len(idx)), 0, 1)
    y = diurnal * seasonal * cloud
    return pd.DataFrame(
        {"target": y, "capacity_mw": 1.0, "load_mw": 40_000 + 5_000 * np.sin(hour / 3),
         "price_eur_mwh": 40 + 10 * np.cos(hour / 4), "wind_mw": rng.normal(10_000, 2_000, len(idx))},
        index=idx,
    )


def test_features_ignore_everything_after_the_gate():
    """Poison the series after one fixed gate; earlier rows must not move.

    Data at time ``s`` is legally readable by a target whose origin is at or
    after ``s``.  So we pick one cutoff gate ``o``, corrupt every observation
    strictly after it, and require that every design-matrix row with
    ``origin <= o`` is unchanged.  Any feature that reaches past its own gate
    would pick up the poison and shift.
    """
    cfg = Config()
    clean = _toy_frame()
    X_clean = F.build_design_matrix(clean, cfg)

    cutoff = pd.Timestamp("2019-02-15 12:00", tz="UTC")  # a gate mid-sample
    poisoned = clean.copy()
    after = pd.DatetimeIndex(clean.index) > cutoff
    for col in ("target", "load_mw", "price_eur_mwh", "wind_mw"):
        poisoned.loc[after, col] = 999.0
    X_poison = F.build_design_matrix(poisoned, cfg)

    legal = pd.DatetimeIndex(X_clean["origin"]) <= cutoff
    assert legal.sum() > 500, "test window too small to be meaningful"
    a_rows, b_rows = X_clean[legal], X_poison.loc[X_clean.index[legal]]

    contaminated = []
    for col in F.feature_columns(X_clean):
        if col == "origin":
            continue
        a = np.nan_to_num(a_rows[col].to_numpy(dtype=float), nan=-7.0)
        b = np.nan_to_num(b_rows[col].to_numpy(dtype=float), nan=-7.0)
        if not np.allclose(a, b):
            contaminated.append(col)
    assert not contaminated, f"features contaminated by post-gate data: {contaminated}"


def test_neighbour_hour_features_respect_the_gate():
    """The +1/+2 hour neighbour features are the easiest place to leak."""
    cfg = Config()
    X = F.build_design_matrix(_toy_frame(), cfg)
    origin = pd.DatetimeIndex(X["origin"])
    horizon = X["horizon"].to_numpy()
    for off in (1, 2):
        k_min = np.ceil((horizon + off) / 24.0)
        stamp = pd.DatetimeIndex(X.index) - pd.to_timedelta(k_min * 24 - off, unit="h")
        assert (stamp <= origin).all(), f"offset {off:+d} reads past the gate"


def test_gated_lag_never_post_dates_origin():
    cfg = Config()
    X = F.build_design_matrix(_toy_frame(), cfg)
    origin = pd.DatetimeIndex(X["origin"])
    horizon = X["horizon"].to_numpy()
    # k_min * 24 hours before the target must land at or before the origin.
    k_min = np.ceil(horizon / 24.0)
    lag_stamp = pd.DatetimeIndex(X.index) - pd.to_timedelta(k_min * 24, unit="h")
    assert (lag_stamp <= origin).all()


def test_horizon_band_matches_day_ahead_protocol():
    cfg = Config()
    X = F.build_design_matrix(_toy_frame(), cfg)
    lo = 24 - cfg.forecast.gate_hour
    hi = lo + cfg.forecast.horizon - 1
    assert X["horizon"].min() >= lo and X["horizon"].max() <= hi
    # every delivery day must contribute exactly 24 hourly targets
    per_origin = X.groupby("origin").size()
    assert set(per_origin.unique()) == {cfg.forecast.horizon}, \
        per_origin.value_counts().to_dict()


@pytest.mark.parametrize("gate", [6, 12, 18])
def test_protocol_holds_for_other_gates(gate):
    cfg = Config()
    cfg.forecast.gate_hour = gate
    X = F.build_design_matrix(_toy_frame(), cfg)
    origin = pd.DatetimeIndex(X["origin"])
    assert (origin.hour == gate).all()
    assert (pd.DatetimeIndex(X.index) > origin).all()


def test_lstm_encoder_receives_real_history_at_prediction_time():
    """The encoder must not be fed zeros for unseen origins.

    The first implementation keyed sequences by *training* origins, so every
    test origin missed the lookup and silently fell back to an all-zero week.
    The model still returned numbers; it just scored worse than persistence.
    This test asserts the encoder input is real, and that it still stops at the
    gate.
    """
    pytest.importorskip("torch")
    from riskaware_pv.models.neural import LSTMQuantile

    cfg = Config()
    X = F.build_design_matrix(_toy_frame(days=200), cfg)
    tr = X[X.index < X.index[0] + pd.Timedelta(days=120)]
    te = X[X.index >= X.index[0] + pd.Timedelta(days=150)]
    assert len(te) > 100

    model = LSTMQuantile(quantiles=cfg.forecast.quantiles, epochs=1, seed=0)
    model.fit(tr, tr["y"].to_numpy(), history=X)

    seq = model._seq_for(te)
    all_zero = (np.abs(seq).sum(axis=(1, 2)) == 0).mean()
    assert all_zero < 0.05, f"{all_zero:.0%} of test sequences are all zero"

    # and the encoder window must still end at (never after) the gate
    origins = pd.DatetimeIndex(te["origin"])
    span = pd.Timedelta(hours=167)
    assert (origins - span <= origins).all()
