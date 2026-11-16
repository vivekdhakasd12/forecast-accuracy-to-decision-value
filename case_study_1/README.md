# Case Study 1: RiskAware-PV

**Probabilistic day-ahead solar forecasting and risk-aware battery dispatch under concept drift.**

Case Study 1 for *AI for Clean Energy*, finished and submitted. It is the base the Master's
thesis builds on; see the [main README](../README.md) for the thesis.

All commands below are run from the repository root, not from this folder.

---

## The one-sentence version

We build a day-ahead PV forecaster that emits a calibrated *predictive
distribution* rather than a single number, feed that distribution into a
two-stage stochastic dispatch problem for a solar + battery site, and measure
the result in euros, because the question an operator actually asks is not
"what is the MAE?" but "what does this forecast cost me?"

## Why this framing

Most forecasting case studies stop at a leaderboard. That is a problem, because
a forecast is an input to a decision, and the mapping from forecast quality to
decision quality is neither obvious nor linear. Three things fall out of taking
the decision seriously:

1. **Asymmetric costs make the mean the wrong answer.** Being short of your
   committed schedule costs far more than being long. The cost-optimal
   commitment is therefore a *quantile* of the predictive distribution, not its
   mean. A point forecast structurally cannot express this.
2. **Coverage is not free, and it is not automatic.** Quantile regression
   produces intervals that are sharp but invalid: our raw models cover 79% at a
   nominal 90%. Conformal prediction repairs this without distributional
   assumptions.
3. **Storage substitutes for information.** A battery buys ex post the same
   insurance a conservative commitment buys ex ante, so the value of a good
   forecast depends on how much flexibility you already own. We quantify the
   trade-off instead of asserting it.

## Research questions

| | Question | Experiment |
|---|---|---|
| **RQ1** | How much day-ahead skill is available with *no* NWP input, and does conformal calibration deliver valid coverage under seasonal non-stationarity? | `e1_forecasting.py` |
| **RQ2** | Does a better forecast score buy a better decision? What is the euro value of the full predictive distribution over a point forecast? | `e2_dispatch.py` |
| **RQ3** | At an equal retraining budget, does drift-triggered retraining beat calendar retraining? | `e3_drift.py` |
| **RQ4** | What information does the model actually run on, and is attribution the same as necessity? | `e4_explain.py` |
| **RQ5** | A new site has days of history, not years. How much local data is needed before training on it beats borrowing a model from another country? | `e5_transfer.py` |

## Data

[Open Power System Data](https://data.open-power-system-data.org/time_series/),
hourly German time series (CC-BY 4.0): PV generation, wind, load, day-ahead
price, and reported installed capacity, 2016-12 to 2020-09 (33,192 hourly
records over 1,383 complete delivery days).

Two data decisions matter:

- **Capacity normalisation.** Installed German PV grew from 39.7 GW to 50.5 GW
  across the sample. Modelling raw MW would bake in a deterministic upward trend
  that flatters any model with a trend term and would contaminate the
  concept-drift experiment with what is really an accounting change. The target
  is a *capacity factor* in [0, 1].
- **No weather input.** The study is deliberately NWP-free: a small operator
  cannot buy a commercial forecast feed. The deterministic part of PV is
  astronomy, and astronomy is free, so solar geometry and clear-sky irradiance
  are computed analytically (NOAA solar position, Haurwitz and Ineichen-Perez
  clear-sky models) and the learner spends its capacity on the stochastic
  residual.

## The part most likely to be wrong, and how we checked it

A day-ahead study is one careless lag away from reporting skill nobody can
realise. With a 12:00 gate, the obvious `y[t-24]` feature is **illegal** for
every delivery hour after noon, because that observation does not exist yet when
the schedule must be submitted.

So no feature uses a fixed lag. For each target the code derives the forecast
origin and takes the most recent same-hour observation that genuinely exists at
that origin. And rather than trusting that by inspection,
`tests/test_no_leakage.py` **poisons** the series after a chosen gate and asserts
the design matrix is unchanged. That test caught two real bugs during
development: the `+1/+2` neighbour-hour features stepped one hour past the gate,
and the horizon band silently dropped hour 0 of every delivery day. Three more
defects were caught elsewhere and are tabulated in §7 of the report, including an
LSTM that was silently fed an all-zero encoder input for **100%** of test rows
and was detectable only because it scored worse than persistence.

## Results

See [the report](report/) (48 pages, PDF) for the full write-up. Headlines:

- Conformal calibration lifts the boosted model's 90% interval coverage from
  **0.70 → 0.81** and cuts its Winkler interval score by 18% (by 57% for the
  worst-calibrated model, which covered just **22.5%** raw). The residual gap is
  seasonal shift breaking exchangeability, which **adaptive** conformal then
  closes: worst-month coverage **0.81 → 0.88**, with *sharper* intervals.
- CRPS and dispatch regret correlate at **r = 0.88** across 20 model variants (a
  figure inflated by including Prophet at the far end of both axes),
  strong enough to be useful, loose enough that individual pairs invert.
  Forecast rank is a good proxy for decision rank, not a substitute.
- Drift-triggered retraining beats calendar retraining on every metric while
  spending **2 refits instead of 7**.
- The value of the predictive distribution over a point forecast ranges from
  **3% to 67%** of dispatch regret depending on imbalance penalty and storage,
  so any single "probabilistic beats point" number is meaningless without them.
- The empirically optimal commitment quantile is **0.55**, meaningfully *less*
  conservative than the **0.667** newsvendor fractile implied by the penalty
  asymmetry, because the battery absorbs ex post what the commitment would
  otherwise have to hedge ex ante.
- SHAP attributes **67.3%** of influence to solar geometry, yet **removing that
  entire family does not hurt accuracy**, while "recent cloud state" carries only
  8.5% of attribution but is the most irreplaceable family. Attribution is not
  necessity.
- **Borrowing beats collecting.** A model trained on Germany and applied unchanged
  to Switzerland, Denmark or Great Britain beats a locally trained model until the
  new site has **120–240 days** of its own history. Pooling source and local data
  is better still: at two of three sites, pooled training on **7 days** of local
  data beats local-only training on **480 days**.
- **Prophet finishes last** (−60% skill vs persistence): a univariate decomposition
  cannot see the recent cloud state, which is the only irreplaceable signal here.
  **LightGBM and XGBoost are indistinguishable** (identical MAE to 4 dp).

## Layout

```
src/riskaware_pv/
  config.py              every tunable number, in one place
  data/
    opsd.py              ingest, QC, capacity/quantile normalisation, multi-zone
    solar_geometry.py    NOAA solar position + clear-sky models (physics, no data)
    features.py          leakage-safe day-ahead design matrix
  models/
    base.py              shared contract, target-space handling, night clamping
    baselines.py         persistence, climatology, ridge, Prophet (all probabilistic)
    gbm.py               boosted trees (LightGBM + XGBoost): point + pinball quantile
    neural.py            LSTM encoder with a monotone quantile head
    conformal.py         split CQR (Mondrian, per horizon) + adaptive conformal
  evaluation/
    metrics.py           MAE/RMSE, pinball, CRPS, PICP/PINAW, Winkler, reliability
    backtest.py          rolling-origin folds with a held-out calibration slice
  decision/
    dispatch.py          two-stage stochastic LP (HiGHS), CVaR via Rockafellar-Uryasev
    policies.py          oracle / point / quantile / stochastic / CVaR commitments
  drift/
    detectors.py         ADWIN, Page-Hinkley, PSI, KS
    policies.py          static vs periodic vs drift-triggered, budget-matched
  explain/
    shap_analysis.py     TreeSHAP, grouped into feature families
    error_decomposition.py  error by horizon, hour, regime; concentration
  transfer.py            local / zero-shot / fine-tune / pooled strategies
case_study_1/experiments/  e1..e5   one script per research question
tests/                   leakage, dispatch economics, scoring rules
```

## Reproducing

```bash
make setup      # venv + dependencies (macOS also needs: brew install libomp)
make data       # download OPSD (~130 MB) and build the design matrix
make test       # 22 tests: leakage, dispatch economics, scoring rules, detectors
make cs1-all    # every experiment, ~45 minutes on a laptop
```

Individual stages: `make cs1-e1` … `make cs1-e5`. Results land in
`case_study_1/results/` as CSV and `case_study_1/figures/` as PNG. Everything is seeded; the
config lives in `configs/default.yaml`.

## Honest limitations

- **No NWP is a self-imposed constraint.** Adding numerical weather prediction
  would improve day-ahead PV skill substantially. The point here is to establish
  what is achievable without it, not to claim it is optimal.
- **National aggregate, single site.** German totals are smoother than any real
  installation, so absolute errors are optimistic for a single rooftop; the
  *relative* comparisons between models and policies are unaffected.
- **Load is treated as known.** This isolates PV uncertainty. A full study would
  add load and price uncertainty, which would raise every policy's regret and
  probably widen the gap in favour of the stochastic commitment.
- **Prices are a synthetic but realistic tariff**, not settled market data, so
  euro figures are indicative of magnitude rather than exact.
- **Not everything in the brief is implemented.** TCN and TFT are genuine gaps.
  Metaheuristics are a deliberate substitution: the dispatch problem is a linear
  program that HiGHS solves to a certified global optimum in 0.16 s, so a
  metaheuristic could only approximate what is already solved exactly.
- **Only Germany's reported capacity is coherent** with its own generation series
  (Switzerland, Denmark and Great Britain report peak generation above nameplate
  by 2.94×, 1.81× and 1.37×), so the transfer study normalises every site by the
  99.9th percentile of its own output instead.
- **Three backtest folds** is few. The leaderboard reports fold-to-fold standard
  errors precisely because neighbouring rows are often not distinguishable.
