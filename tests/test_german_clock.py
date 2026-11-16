"""The real German day-ahead clock (configs/thesis.yaml).

* The gate is 12:00 German time: 11:00 UTC in winter, 10:00 UTC in summer.
* A delivery day is a German calendar day: 23 hours on the last Sunday of
  March, 25 on the last Sunday of October, 24 otherwise.
* Data is stamped with the start of its hour, so at a 12:00 gate the newest
  complete value is the one stamped 11:00.  Nothing stamped 12:00 or later
  may reach a feature.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from riskaware_pv.config import ROOT, Config
from riskaware_pv.data import features as F

TZ = "Europe/Berlin"


def _berlin_cfg() -> Config:
    cfg = copy.deepcopy(Config())
    cfg.forecast.timezone = TZ
    cfg.forecast.availability_lag_hours = 1
    return cfg


def _toy_frame(start: str = "2021-02-01", days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    diurnal = np.clip(np.sin((hour - 5) / 12 * np.pi), 0, None)
    seasonal = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    y = diurnal * seasonal * np.clip(rng.beta(4, 2, len(idx)), 0, 1)
    return pd.DataFrame({
        "target": y, "capacity_mw": 1.0,
        "load_mw": 50_000 + 5_000 * np.sin(hour / 3),
        "price_eur_mwh": 60 + 20 * np.cos(hour / 4),
    }, index=idx)


def test_thesis_config_uses_the_german_clock():
    cfg = Config.load(ROOT / "configs" / "thesis.yaml")
    assert cfg.forecast.timezone == TZ
    assert cfg.forecast.availability_lag_hours == 1
    assert Config.load().forecast.timezone == "UTC"        # Case Study 1 untouched


def test_gate_is_noon_german_time():
    winter = pd.DatetimeIndex([pd.Timestamp("2024-01-16 09:00", tz="UTC")])
    summer = pd.DatetimeIndex([pd.Timestamp("2024-07-16 09:00", tz="UTC")])
    assert F.forecast_origin(winter, 12, TZ)[0] == pd.Timestamp("2024-01-15 11:00", tz="UTC")
    assert F.forecast_origin(summer, 12, TZ)[0] == pd.Timestamp("2024-07-15 10:00", tz="UTC")


def test_hour_just_after_german_midnight_belongs_to_the_new_day():
    # 23:30 UTC on 15 Jan is 00:30 German time on 16 Jan -> decided on 15 Jan.
    t = pd.DatetimeIndex([pd.Timestamp("2024-01-15 23:00", tz="UTC")])
    assert F.forecast_origin(t, 12, TZ)[0] == pd.Timestamp("2024-01-15 11:00", tz="UTC")


def test_clock_change_days_have_23_and_25_hours():
    origins = pd.DatetimeIndex([
        pd.Timestamp("2021-03-27 11:00", tz="UTC"),   # decides Sun 28 Mar (spring)
        pd.Timestamp("2021-06-10 10:00", tz="UTC"),   # an ordinary summer day
        pd.Timestamp("2021-10-30 10:00", tz="UTC"),   # decides Sun 31 Oct (autumn)
    ])
    assert F.delivery_day_hours(origins, TZ).tolist() == [23, 24, 25]


def test_design_matrix_keeps_short_and_long_days():
    X = F.build_design_matrix(_toy_frame(), _berlin_cfg())
    sizes = X.groupby("origin").size()
    assert set(sizes.unique()) == {23, 24, 25}
    # every row is delivered on the German day after its own gate
    local_day = pd.DatetimeIndex(X.index).tz_convert(TZ).normalize().tz_localize(None)
    gate_day = pd.DatetimeIndex(X["origin"]).tz_convert(TZ).normalize().tz_localize(None)
    assert ((local_day - gate_day) == pd.Timedelta(days=1)).all()


@pytest.mark.parametrize("gate", ["2021-03-10 11:00", "2021-08-10 10:00"])
def test_nothing_from_the_gate_hour_onwards_leaks(gate):
    """Poison every value stamped at or after the gate; earlier rows must not move."""
    cfg = _berlin_cfg()
    clean = _toy_frame()
    X_clean = F.build_design_matrix(clean, cfg)
    g = pd.Timestamp(gate, tz="UTC")
    poisoned = clean.copy()
    poisoned.loc[poisoned.index >= g, ["target", "load_mw", "price_eur_mwh"]] = 999.0
    X_poison = F.build_design_matrix(poisoned, cfg)

    legal = pd.DatetimeIndex(X_clean["origin"]) <= g
    assert legal.sum() > 500, "test window too small to be meaningful"
    cols = [c for c in F.feature_columns(X_clean) if c != "origin"]
    a = X_clean.loc[legal, cols].to_numpy(dtype=float)
    b = X_poison.loc[X_clean.index[legal], cols].to_numpy(dtype=float)
    assert np.allclose(np.nan_to_num(a, nan=-7), np.nan_to_num(b, nan=-7))


def test_the_last_complete_hour_is_actually_used():
    """The cutoff is not needlessly conservative: the hour that ends exactly at
    the gate (stamped gate - 1h) does reach the features."""
    cfg = _berlin_cfg()
    clean = _toy_frame()
    X_clean = F.build_design_matrix(clean, cfg)
    g = pd.Timestamp("2021-03-10 11:00", tz="UTC")
    poisoned = clean.copy()
    poisoned.loc[g - pd.Timedelta(hours=1), "target"] = 999.0
    X_poison = F.build_design_matrix(poisoned, cfg)
    rows = X_clean.index[pd.DatetimeIndex(X_clean["origin"]) == g]
    assert not np.allclose(X_clean.loc[rows, "y_at_origin"], X_poison.loc[rows, "y_at_origin"])
