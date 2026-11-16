"""Correctness checks for the scoring rules and the conformal wrappers.

Metrics are the instruments of the study; if they are wrong, every conclusion
is wrong in a way no amount of modelling care can detect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskaware_pv.evaluation import metrics as M
from riskaware_pv.models.conformal import conformal_quantile


def test_pinball_is_minimised_at_the_true_quantile():
    """The defining property of the pinball loss."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=20_000)
    q = 0.8
    truth = np.quantile(y, q)
    candidates = truth + np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    losses = [M.pinball_loss(y, np.full((len(y), 1), c), [q]) for c in candidates]
    assert int(np.argmin(losses)) == 2, losses


def test_crps_of_a_perfect_deterministic_forecast_is_zero():
    y = np.linspace(0, 1, 200)
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q_pred = np.repeat(y[:, None], len(qs), axis=1)
    assert M.crps_from_quantiles(y, q_pred, qs) == pytest.approx(0.0, abs=1e-12)


def test_crps_prefers_the_sharper_of_two_calibrated_forecasts():
    """Both forecasts are centred on the truth; the tighter one must win."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=5000)
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    from scipy.stats import norm
    tight = y[:, None] * 0 + norm.ppf(qs)[None, :] * 1.0
    wide = y[:, None] * 0 + norm.ppf(qs)[None, :] * 3.0
    assert M.crps_from_quantiles(y, tight, qs) < M.crps_from_quantiles(y, wide, qs)


def test_winkler_punishes_a_wide_interval_that_picp_would_reward():
    """PICP is gameable, the interval score is not -- that is why we report both."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=5000)
    honest_lo, honest_hi = np.full(5000, -1.645), np.full(5000, 1.645)
    lazy_lo, lazy_hi = np.full(5000, -10.0), np.full(5000, 10.0)
    assert M.picp(y, lazy_lo, lazy_hi) > M.picp(y, honest_lo, honest_hi)
    assert M.winkler_score(y, lazy_lo, lazy_hi, 0.1) > \
           M.winkler_score(y, honest_lo, honest_hi, 0.1)


def test_conformal_quantile_gives_finite_sample_coverage():
    """The (n+1)/n correction must not under-cover on small calibration sets."""
    rng = np.random.default_rng(3)
    alpha = 0.1
    hits = []
    for _ in range(400):
        cal = rng.normal(size=60)
        new = rng.normal()
        thresh = conformal_quantile(cal, alpha)
        hits.append(new <= thresh)
    # marginal coverage must be at least 1 - alpha (conformal guarantee)
    assert np.mean(hits) >= 1 - alpha - 0.03, np.mean(hits)


def test_reliability_curve_is_exact_for_a_known_distribution():
    rng = np.random.default_rng(4)
    y = rng.uniform(size=20_000)
    qs = np.array([0.1, 0.5, 0.9])
    q_pred = np.repeat(qs[None, :], len(y), axis=0)  # true quantiles of U(0,1)
    curve = M.reliability_curve(y, q_pred, qs)
    assert np.allclose(curve["observed"], curve["nominal"], atol=0.02)


def test_conformal_wrapper_regrids_to_a_different_quantile_set():
    """Asking for 3 levels from a wrapper calibrated on 7 must interpolate, not raise."""
    from riskaware_pv.config import Config
    from riskaware_pv.data import features as F
    from riskaware_pv.models.baselines import SeasonalPersistence
    from riskaware_pv.models.conformal import ConformalizedQuantile
    from tests.test_no_leakage import _toy_frame

    cfg = Config()
    X = F.build_design_matrix(_toy_frame(days=140), cfg)
    tr, cal, te = X.iloc[:1500], X.iloc[1500:2200], X.iloc[2200:]
    model = SeasonalPersistence().fit(tr, tr["y"].to_numpy())
    cq = ConformalizedQuantile(model, cfg.forecast.quantiles, condition_on="horizon")
    cq.calibrate(cal, cal["y"].to_numpy())

    out = cq.predict_quantiles(te, np.array([0.05, 0.5, 0.95]))
    assert out.shape == (len(te), 3)
    assert np.all(np.diff(out, axis=1) >= -1e-9), "regridded quantiles must stay ordered"


def test_detectors_are_scale_free_after_standardisation():
    """The same shift must be detected identically at any signal magnitude.

    Without standardisation ADWIN's absolute Hoeffding bound silently ignores
    small-magnitude streams -- which is exactly why the drift experiment
    reported zero triggers on its first run, where the daily-MAE stream has
    magnitude ~0.04.  The stream is generated once and merely rescaled, so any
    difference in output is attributable to scale alone.
    """
    import pandas as pd
    from riskaware_pv.drift.detectors import ADWINDetector, PageHinkley

    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=800, freq="D")
    base = np.concatenate([rng.normal(1.0, 0.3, 400), rng.normal(3.0, 0.3, 400)])

    for make in (lambda: ADWINDetector(0.002, warmup=90), lambda: PageHinkley(warmup=90)):
        fired = {}
        for scale in (10.0, 0.02):
            events = make().run(pd.Series(base * scale, index=idx))
            assert events, f"{make().name} missed a 3x shift at scale {scale}"
            fired[scale] = [e.index for e in events]
        assert fired[10.0] == fired[0.02], (
            f"{make().name} is not scale-invariant: {fired}")
        assert any(380 <= i <= 560 for i in fired[10.0]), (
            f"{make().name} never fired near the true change point: {fired[10.0]}")


def test_adaptive_conformal_can_widen_past_the_base_interval():
    """ACI must not be capped by the fitted quantile grid.

    The score-based formulation is what makes this true; the earlier
    quantile-level formulation was structurally unable to exceed the model's own
    5-95% band.
    """
    from riskaware_pv.models.conformal import AdaptiveConformal

    class _Stub:
        name = "stub"
        def predict_quantiles(self, X, quantiles):
            return np.repeat(np.array([[-0.1, 0.1]]), len(X), axis=0)
        def _postprocess(self, X, p):
            return p

    n = 600
    X = pd.DataFrame({"horizon": np.arange(n) % 24})
    y_cal = np.random.default_rng(0).normal(0, 1.0, 300)
    X_cal = pd.DataFrame({"horizon": np.arange(300) % 24})
    aci = AdaptiveConformal(_Stub(), alpha=0.1, gamma=0.05).seed(X_cal, y_cal)
    traj = aci.run(X, np.random.default_rng(1).normal(0, 1.0, n))
    base_width = 0.2
    assert (traj["upper"] - traj["lower"]).mean() > base_width * 2, \
        "ACI never widened beyond the base model's interval"
    assert traj["miss"].mean() < 0.25, "ACI failed to approach nominal coverage"


def test_openmp_is_serialised_before_the_libraries_load():
    """Guard the fix for the LightGBM + torch deadlock.

    Both libraries ship their own OpenMP runtime; with more than one thread the
    combination hangs on macOS with no traceback.  If this default is ever
    relaxed, the forecasting experiment silently stops making progress, so the
    setting is asserted rather than left to a comment.
    """
    import os
    import riskaware_pv  # noqa: F401  (import configures the environment)

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
