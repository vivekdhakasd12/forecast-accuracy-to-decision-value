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
# Case Study 1 outputs (experiments e1..e5) stay with the case study.
CASE_STUDY_1 = ROOT / "case_study_1"
RESULTS = CASE_STUDY_1 / "results"
FIGURES = CASE_STUDY_1 / "figures"
# Thesis outputs get their own folders so the two never overwrite each other.
THESIS = ROOT / "thesis"
THESIS_RESULTS = THESIS / "results"
THESIS_FIGURES = THESIS / "figures"

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

    At ``gate_hour`` on day D (clock time in ``timezone``) we must submit a
    forecast for every hour of delivery day D+1, a calendar day in the same
    ``timezone``.  No feature may use an observation that is not complete by
    the gate.

    ``timezone``
        ``"UTC"`` reproduces Case Study 1: gate at 12:00 UTC, 24-hour UTC days.
        ``"Europe/Berlin"`` is the real German market: gate at 12:00 German
        time, delivery days of 23, 24 or 25 hours around the clock changes.
    ``availability_lag_hours``
        How long after its timestamp an observation becomes usable.  SMARD and
        OPSD stamp each hour with its *start*, so the value stamped 11:00 covers
        11:00-12:00 and is complete at 12:00: a lag of 1.  Case Study 1 used 0,
        which let the hour that starts at the gate count as known.
    """

    gate_hour: int = 12
    horizon: int = 24              # hours in a normal delivery day
    timezone: str = "UTC"
    availability_lag_hours: int = 0
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
    # Grid connection limits.  None derives them from load, PV and battery
    # power (see dispatch.grid_limits); they never bind under the Case Study 1
    # tariff and exist so that negative real prices cannot make the LP unbounded.
    grid_import_max_mw: float | None = None
    grid_export_max_mw: float | None = None


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
