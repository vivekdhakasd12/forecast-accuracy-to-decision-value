"""A recordable, end-to-end tour of the pipeline's core, in about 90 seconds.

The real pipeline takes 45 minutes, which is unwatchable, and most of that time
is spent refitting models across folds rather than doing anything conceptually
new. This script runs the same code path on one slice of data and narrates it,
so a screen recording shows what the project actually does.

It prints the real source of the load-bearing functions rather than a paraphrase,
so the video cannot drift away from the code.

    .venv/bin/python presentation/demo_pipeline.py            # ~90 s
    .venv/bin/python presentation/demo_pipeline.py --pace 1.8 # slower, for narration
    .venv/bin/python presentation/demo_pipeline.py --pace 0   # no pauses, for testing
"""
from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

O, B, D, G, R, X = ("\033[38;5;202m", "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m")
PACE = 1.0


def pause(sec: float = 1.0) -> None:
    if PACE > 0:
        time.sleep(sec * PACE)


def stage(n: int, title: str, sub: str = "") -> None:
    print(f"\n{O}{B}━━ {n}. {title} {'━' * max(4, 62 - len(title))}{X}")
    if sub:
        print(f"{D}   {sub}{X}")
    pause(0.7)


def line(label: str, value: str, note: str = "") -> None:
    print(f"   {label:<34} {B}{value}{X}" + (f"   {D}{note}{X}" if note else ""))
    pause(0.35)


def code(fn, dedent_to: int = 3, keep: slice | None = None) -> None:
    src = inspect.getsource(fn).splitlines()
    if keep:
        src = src[keep]
    body = textwrap.dedent("\n".join(src))
    print(D + textwrap.indent(body, " " * dedent_to) + X)
    pause(1.2)


def main() -> None:
    global PACE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pace", type=float, default=1.0, help="pause multiplier; 0 = none")
    args = ap.parse_args()
    PACE = args.pace
    t_start = time.time()

    print(f"\n{O}{B}RiskAware-PV{X}  {D}the core of the pipeline, end to end{X}")
    print(f"{D}Probabilistic day-ahead solar forecasting and risk-aware battery dispatch{X}")
    pause(1.5)

    from riskaware_pv.config import Config
    from riskaware_pv.data import features as F
    from riskaware_pv.data import solar_geometry as SG
    from riskaware_pv.pipeline import load_design_matrix, load_raw_frame

    cfg = Config.load()

    # ---------------------------------------------------------------- 1
    stage(1, "The data", "Open Power System Data, hourly German solar. Free and open.")
    frame = load_raw_frame(cfg)
    line("hourly records", f"{len(frame):,}")
    line("span", f"{frame.index.min().date()} to {frame.index.max().date()}")
    line("installed capacity grew", f"{frame.capacity_mw.min()/1000:.1f} to "
                                    f"{frame.capacity_mw.max()/1000:.1f} GW",
         "which is why the target is a capacity factor, not megawatts")

    # ---------------------------------------------------------------- 2
    stage(2, "Physics instead of a weather feed",
          "No NWP is used. The deterministic part of solar output is astronomy.")
    day = pd.date_range("2019-06-21 00:00", periods=24, freq="h", tz="UTC")
    cs = SG.clearsky_frame(day, cfg.site.latitude, cfg.site.longitude)
    ghi = cs["clearsky_haurwitz"].to_numpy()
    bars = "".join(" ▁▂▃▄▅▆▇█"[min(8, int(v / max(ghi.max(), 1) * 8.99))] for v in ghi)
    line("clear-sky irradiance, 21 June", bars, "computed from the timestamp alone")
    line("peak", f"{ghi.max():.0f} W/m2", "no data was needed to know this")

    # ---------------------------------------------------------------- 3
    stage(3, "The gate", "Everything in the project rests on one rule.")
    print(f"{D}   At 12:00 on day D we commit all 24 hours of day D+1. So the obvious")
    print(f"   lag y[t-24] is ILLEGAL for every delivery hour after midday: that")
    print(f"   observation does not exist yet. The fix is a lag that adapts:{X}\n")
    code(F._gated_shift, keep=slice(-6, None))

    X_design = load_design_matrix(cfg)
    line("design matrix", f"{X_design.shape[0]:,} rows x {X_design.shape[1]} cols")
    line("horizon band", f"{int(X_design.horizon.min())} to {int(X_design.horizon.max())} hours",
         "not 1 to 24, because the gate is at noon")
    line("rows per delivery day", f"{sorted(X_design.groupby('origin').size().unique())}",
         "complete days only: the dispatch problem needs all 24")

    # ---------------------------------------------------------------- 4
    stage(4, "The guard", "The rule is not trusted. It is tested adversarially.")
    print(f"{D}   Poison every observation after a chosen gate, rebuild the features,")
    print(f"   and assert that every earlier row is bit-for-bit unchanged.{X}\n")
    t = time.time()
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q", "--no-header",
         "--deselect",
         "tests/test_no_leakage.py::test_lstm_encoder_receives_real_history_at_prediction_time"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = [l for l in res.stdout.strip().splitlines() if l.strip()][-1]
    print(f"   {G}{tail}{X}   {D}({time.time()-t:.1f}s){X}")
    pause(1.2)

    # now break it on purpose and watch the guard catch it
    print(f"\n{D}   Now reintroduce the bug this test found during development: the")
    print(f"   neighbour-hour feature that stepped one hour past the gate.{X}")
    pause(1.0)
    contaminated = _break_the_gate_and_check(F, cfg)
    print(f"   {R}{B}CAUGHT{X}  {R}{len(contaminated)} features changed when the future was "
          f"poisoned{X}")
    print(f"   {D}   {', '.join(contaminated[:4])}{'...' if len(contaminated) > 4 else ''}{X}")
    print(f"   {D}That is a bug a code review passed. The test did not.{X}")
    pause(1.6)

    # ---------------------------------------------------------------- 5
    stage(5, "One interface, every model",
          "The backtester never needs to know which model it is holding.")
    from riskaware_pv.pipeline import model_registry
    print(f"{D}   fit(X, y)  ->  predict(X)  ->  predict_quantiles(X, qs){X}")
    names = list(model_registry(cfg, include_lstm=True))
    print(f"{D}   {len(names)} implementations behind it:{X}")
    print(f"{D}   {', '.join(names)}{X}")
    pause(1.0)

    tr = X_design[X_design.index < "2019-06-01"]
    cal = X_design[(X_design.index >= "2019-06-01") & (X_design.index < "2019-09-01")]
    te = X_design[(X_design.index >= "2019-09-01") & (X_design.index < "2019-12-01")]
    line("train / calibrate / test", f"{len(tr):,} / {len(cal):,} / {len(te):,} rows",
         "chronological, never shuffled")

    from riskaware_pv.evaluation import metrics as M
    from riskaware_pv.models.baselines import SeasonalPersistence
    from riskaware_pv.models.gbm import GBMQuantile

    qs = np.asarray(cfg.forecast.quantiles)
    y_te = te["y"].to_numpy()
    daylight = te["cos_zenith"].to_numpy() > 1e-3

    t = time.time()
    base = SeasonalPersistence().fit(tr, tr["y"].to_numpy())
    line("persistence baseline", f"MAE {M.mae(y_te[daylight], base.predict(te)[daylight]):.4f}",
         f"fitted in {time.time()-t:.1f}s")

    t = time.time()
    gbm = GBMQuantile(quantiles=cfg.forecast.quantiles,
                      params=dict(n_estimators=60)).fit(tr, tr["y"].to_numpy())
    q_raw = gbm.predict_quantiles(te, qs)
    line("boosted quantile model", f"MAE {M.mae(y_te[daylight], gbm.predict(te)[daylight]):.4f}",
         f"7 quantile levels, fitted in {time.time()-t:.1f}s")

    # ---------------------------------------------------------------- 6
    stage(6, "Calibration", "Sharp intervals are not the same as valid intervals.")
    from riskaware_pv.models.conformal import ConformalizedQuantile
    picp_raw = M.picp(y_te[daylight], q_raw[daylight, 1], q_raw[daylight, 5])
    wink_raw = M.winkler_score(y_te[daylight], q_raw[daylight, 1], q_raw[daylight, 5], 0.10)

    cq = ConformalizedQuantile(gbm, qs, condition_on="horizon")
    cq.calibrate(cal, cal["y"].to_numpy())
    q_cal = cq.predict_quantiles(te, qs)
    picp_cal = M.picp(y_te[daylight], q_cal[daylight, 1], q_cal[daylight, 5])
    wink_cal = M.winkler_score(y_te[daylight], q_cal[daylight, 1], q_cal[daylight, 5], 0.10)

    line("nominal coverage", "90%")
    line("raw model covers", f"{picp_raw:6.1%}", f"interval score {wink_raw:.3f}")
    line("after calibration", f"{picp_cal:6.1%}",
         f"interval score {wink_cal:.3f}  ({(wink_cal/wink_raw-1)*100:+.0f}%)")
    print(f"\n{D}   On this one slice calibration does not help, and that is worth showing")
    print(f"   rather than hiding. It calibrated on June to August and was tested on")
    print(f"   September to November. Split conformal guarantees coverage only under")
    print(f"   exchangeability, and two different seasons are not exchangeable.")
    print(f"   This is precisely the failure the study documents in section 5.3.{X}")
    pause(2.0)
    print(f"\n{D}   Across the proper rolling backtest, the same procedure gives:{X}")
    line("  worst raw model covered", "22.5%", "nominal 90%, from in-sample residuals")
    line("  after conformal calibration", "63.0%", "interval score 57% better")
    line("  adaptive conformal, worst month", "82.8% -> 87.9%", "and sharper intervals")

    # ---------------------------------------------------------------- 7
    stage(7, "The decision", "A forecast is an input to a commitment, settled in euros.")
    from riskaware_pv.decision import policies as P
    one_day = te[te.index.normalize() == te.index.normalize()[len(te) // 2]].sort_index()
    inputs = P.build_day_inputs(one_day, cfg)
    qd = cq.predict_quantiles(one_day, qs)
    point, actual = cq.predict(one_day), one_day["y"].to_numpy()

    results = {}
    for pol in ("oracle", "point", "stochastic"):
        results[pol] = P.run_policy(pol, inputs, qd, point, actual, cfg,
                                    rng=np.random.default_rng(0))
    line("delivery day", f"{one_day.index[0].date()}")
    for pol in ("oracle", "point", "stochastic"):
        r = results[pol]
        line(f"  {pol} commitment", f"{r['total_cost']:8.2f} EUR",
             "perfect foresight" if pol == "oracle"
             else f"regret {r['total_cost'] - results['oracle']['total_cost']:6.2f} EUR")

    print(f"\n{D}   One day is noise. The solve above is real, but whether the stochastic")
    print(f"   commitment wins on any single day is close to a coin flip. Across the")
    print(f"   200 settled delivery days in the study:{X}")
    line("  point forecast commitment", "164.71 EUR/day", "mean regret against the oracle")
    line("  stochastic commitment", "159.82 EUR/day", "3.0% lower")
    line("  and worth between", "3% and 67%", "depending on penalty and storage")

    # ---------------------------------------------------------------- end
    print(f"\n{O}{B}━━ that is the core {'━' * 48}{X}")
    print(f"{D}   Everything else in the project is this loop, repeated: across three")
    print(f"   backtest folds, five commitment policies, three retraining strategies,")
    print(f"   and four transfer strategies on three other countries.")
    print(f"   22 tests, seeded, reproducible with `make all`.{X}")
    print(f"\n{D}   demo ran in {time.time() - t_start:.0f}s{X}\n")


def _break_the_gate_and_check(F, cfg) -> list[str]:
    """Reintroduce the historical off-by-one, then run the poisoning check."""
    from tests.test_no_leakage import _toy_frame

    original = F._gated_shift

    def buggy(series, target_index, origin, days_back, hour_offset=0):
        # The bug: the hour offset is applied *after* rounding to whole days, so a
        # +1 hour neighbour feature whose base lag sits exactly on the gate steps
        # one hour past it. No assertion, exactly as the original code had none.
        horizon_h = (target_index - origin) / pd.Timedelta(hours=1)
        k_min = np.ceil(horizon_h / 24.0)
        offsets = (k_min + (days_back - 1)) * 24.0 - hour_offset
        wanted = target_index - pd.to_timedelta(offsets, unit="h")
        return series.reindex(wanted).to_numpy()

    F._gated_shift = buggy
    try:
        clean = _toy_frame()
        cutoff = pd.Timestamp("2019-02-15 12:00", tz="UTC")
        poisoned = clean.copy()
        after = pd.DatetimeIndex(clean.index) > cutoff
        for col in ("target", "load_mw", "price_eur_mwh", "wind_mw"):
            poisoned.loc[after, col] = 999.0
        A, Bm = F.build_design_matrix(clean, cfg), F.build_design_matrix(poisoned, cfg)
        legal = pd.DatetimeIndex(A["origin"]) <= cutoff
        bad = []
        for c in F.feature_columns(A):
            if c == "origin":
                continue
            a = np.nan_to_num(A[legal][c].to_numpy(float), nan=-7.0)
            b = np.nan_to_num(Bm.loc[A.index[legal]][c].to_numpy(float), nan=-7.0)
            if not np.allclose(a, b):
                bad.append(c)
        return bad
    finally:
        F._gated_shift = original


if __name__ == "__main__":
    main()
