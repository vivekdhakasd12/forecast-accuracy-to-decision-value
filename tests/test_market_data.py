"""Offline tests for the SMARD and reBAP loaders (no network access needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskaware_pv.config import Config
from riskaware_pv.data import features as F
from riskaware_pv.data import imbalance as IMB
from riskaware_pv.data import smard as SM

# The example from the Netztransparenz Web-API documentation (format 9).
REBAP_SAMPLE = """Datum;Zeitzone;von;bis;Datenkategorie;Datentyp;Einheit;reBAP unterdeckt;reBAP ueberdeckt
10.06.2023;UTC;13:00;13:15;reBAP;Qualitätsgesichert;EUR/MWh;103,97;103,97
10.06.2023;UTC;13:15;13:30;reBAP;Qualitätsgesichert;EUR/MWh;104,06;104,06
10.06.2023;UTC;13:30;13:45;reBAP;Qualitätsgesichert;EUR/MWh;104,57;104,57
10.06.2023;UTC;13:45;14:00;reBAP;Qualitätsgesichert;EUR/MWh;-1.234,50;-1.234,50
10.06.2023;UTC;14:00;14:15;reBAP;Qualitätsgesichert;EUR/MWh;50,00;50,00
"""


def test_parse_rebap_csv():
    df = IMB.parse_rebap_csv(REBAP_SAMPLE)
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2023-06-10 13:00", tz="UTC")
    assert df.index[3] == pd.Timestamp("2023-06-10 13:45", tz="UTC")
    assert df["rebap_short_eur_mwh"].iloc[0] == pytest.approx(103.97)
    assert df["rebap_short_eur_mwh"].iloc[3] == pytest.approx(-1234.50)
    assert (df["rebap_short_eur_mwh"] == df["rebap_long_eur_mwh"]).all()


def test_rebap_hourly_needs_four_quarter_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(IMB, "RAW_DIR", tmp_path)
    monkeypatch.delenv("NTP_CLIENT_ID", raising=False)
    monkeypatch.delenv("NTP_CLIENT_SECRET", raising=False)
    (tmp_path / "sample.csv").write_text(REBAP_SAMPLE, encoding="utf-8")
    hourly = IMB.load_rebap_hourly("2023-06-10 13:00", "2023-06-10 14:00")
    full = hourly.loc[pd.Timestamp("2023-06-10 13:00", tz="UTC")]
    assert full == pytest.approx((103.97 + 104.06 + 104.57 - 1234.50) / 4)
    # 14:00 has one quarter hour only: not a price
    assert np.isnan(hourly.loc[pd.Timestamp("2023-06-10 14:00", tz="UTC")])


def test_rebap_without_data_or_credentials_says_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(IMB, "RAW_DIR", tmp_path / "empty")
    monkeypatch.delenv("NTP_CLIENT_ID", raising=False)
    monkeypatch.delenv("NTP_CLIENT_SECRET", raising=False)
    with pytest.raises(FileNotFoundError, match="NTP_CLIENT_ID"):
        IMB.load_rebap("2023-01-01", "2023-01-31")


def test_parse_smard_series():
    payload = {"meta_data": {}, "series": [[1688371200000, 10.5], [1688374800000, None],
                                           [1688378400000, -3.0]]}
    s = SM.parse_smard_series(payload)
    assert s.index[0] == pd.Timestamp("2023-07-03 08:00", tz="UTC")
    assert s.iloc[0] == 10.5 and np.isnan(s.iloc[1]) and s.iloc[2] == -3.0


def test_parse_capacity_in_mw():
    payload = {"time": ["01.2019", "02.2019"],
               "production_types": [{"name": "Solar AC", "data": [40.0, 41.0]},
                                    {"name": "Solar DC", "data": [46.242, 46.9]}]}
    cap = SM.parse_capacity(payload)
    assert cap.iloc[0] == pytest.approx(46_242.0)
    assert cap.index[1] == pd.Timestamp("2019-02-01", tz="UTC")


def test_settlement_columns_never_reach_the_features():
    """Imbalance prices and the TSO forecast are published after the gate.

    Poison them completely: the design matrix must not move at all.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=60 * 24, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    y = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * rng.beta(4, 2, len(idx))
    frame = pd.DataFrame({
        "target": y, "capacity_mw": 1.0,
        "load_mw": 50_000 + 5_000 * np.sin(hour / 3),
        "price_eur_mwh": 60 + 20 * np.cos(hour / 4),
        "solar_forecast_tso_mw": y * 40_000,
        "imbalance_price_eur_mwh": rng.normal(80, 50, len(idx)),
    }, index=idx)
    cfg = Config()
    X_clean = F.build_design_matrix(frame, cfg)
    poisoned = frame.copy()
    poisoned["solar_forecast_tso_mw"] = 1e9
    poisoned["imbalance_price_eur_mwh"] = -1e9
    X_poison = F.build_design_matrix(poisoned, cfg)
    pd.testing.assert_frame_equal(X_clean, X_poison)
