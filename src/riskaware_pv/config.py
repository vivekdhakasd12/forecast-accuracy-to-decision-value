"""Central configuration for the RiskAware-PV case study.

Everything that a reviewer might want to change (site, horizon, quantiles,
battery size, backtest windows) lives here or in ``configs/default.yaml`` so
that no experiment hides a magic number in its body.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass
class SiteConfig:
    """Geographic anchor used for the physical clear-sky model."""

    name: str = "DE_transparency"
    latitude: float = 51.0      # approximate load-weighted centroid of Germany
    longitude: float = 10.0
    # Nameplate used only to normalise the series into [0, 1] "capacity factor".
    capacity_mw: float = 45_000.0


@dataclass
class ForecastConfig:
    """Day-ahead protocol.

    At ``gate_hour`` (local/UTC) on day D we must submit a forecast for all 24
    hours of day D+1.  Horizons therefore run from ``gate_hour`` + 1 up to
    ``gate_hour`` + 24 hours, and *no* feature may use an observation newer
    than the gate.  This mirrors European day-ahead market closure.
    """

    gate_hour: int = 12
    horizon: int = 24
    quantiles: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    target: str = "solar"          # "solar" | "wind" | "net_load"
    lags_hours: Sequence[int] = (24, 25, 26, 48, 72, 168, 336)
    rolling_windows: Sequence[int] = (24, 72, 168)


@dataclass
class BacktestConfig:
    """Rolling-origin evaluation."""

    train_years: float = 2.0
    calibration_days: int = 90     # split-conformal calibration slice
    test_days: int = 365
    step_days: int = 30            # refit cadence for the "periodic" policy
    min_train_days: int = 365


@dataclass
class BatteryConfig:
    """Storage asset for the downstream dispatch problem (per-unit of site)."""

    energy_mwh: float = 4.0        # usable energy
    power_mw: float = 1.0          # symmetric charge/discharge limit
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_init: float = 0.5          # fraction of energy_mwh
    soc_min: float = 0.05
    soc_max: float = 0.95
    # Site scale: we model a 1 MW-class prosumer, so the national series is
    # rescaled to this nameplate.
    pv_nameplate_mw: float = 2.0
    load_peak_mw: float = 1.2


@dataclass
class MarketConfig:
    """Prices and penalties driving the dispatch objective (EUR/MWh)."""

    import_price_peak: float = 220.0
    import_price_offpeak: float = 120.0
    export_price: float = 45.0
    # Imbalance penalty applied to any deviation from the day-ahead schedule.
    imbalance_penalty_up: float = 300.0    # short: had to buy at short notice
    imbalance_penalty_down: float = 90.0   # long: spilled / curtailed
    peak_hours: Sequence[int] = (7, 8, 9, 10, 17, 18, 19, 20)
    cvar_alpha: float = 0.90               # CVaR tail level for risk-aware policy
    n_scenarios: int = 40


@dataclass
class DriftConfig:
    # Thresholds apply to a z-scored error stream; see drift.detectors.
    adwin_delta: float = 0.002
    ph_delta: float = 0.05
    ph_lambda: float = 8.0
    psi_bins: int = 10
    psi_threshold: float = 0.20
    retrain_budget: int = 6        # max refits allowed in the test period
    cooldown_days: int = 21        # min days between drift-triggered refits


@dataclass
class Config:
    site: SiteConfig = field(default_factory=SiteConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    seed: int = 20260101

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = ROOT / "configs" / "default.yaml"
        path = Path(path)
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text()) or {}
        sub = {
            "site": SiteConfig, "forecast": ForecastConfig,
            "backtest": BacktestConfig, "battery": BatteryConfig,
            "market": MarketConfig, "drift": DriftConfig,
        }
        kwargs = {k: v(**raw.get(k, {})) for k, v in sub.items()}
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)
