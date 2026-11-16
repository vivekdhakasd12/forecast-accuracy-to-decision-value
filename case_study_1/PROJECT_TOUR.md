> Paths in this tour are relative to `case_study_1/`, except `src/`, `tests/`, `configs/` and `data/`, which sit at the repository root.

# Where everything is

A map of the repository, in the order you would explain it.

| Path | What it is |
|---|---|
| `README.md` | What the project is, the five research questions, headline results |
| `Makefile` | One command per stage. `make all` reproduces everything |
| `configs/default.yaml` | Every tunable number in one place: site, gate hour, quantiles, battery, prices, detector thresholds |
| `src/riskaware_pv/` | The library. Nothing here runs on its own |
| `experiments/` | Five scripts, one per research question. These are what you run |
| `tests/` | 22 tests. The leakage guard lives here |
| `figures/` | Generated. 19 figures, written by the experiments |
| `results/` | Generated. Score tables and per-day costs as CSV |
| `data/` | Downloaded and cached. Rebuilt by `make data` |
| `report/` | The report: LaTeX source, the PDF, and the diagram script |
| `presentation/` | Deck, speaker script, demo video and its build scripts |

## Inside the library

| Module | Responsibility |
|---|---|
| `data/opsd.py` | Download, quality control, normalisation, multi-country loading |
| `data/solar_geometry.py` | Sun position and clear-sky irradiance, computed from physics |
| `data/features.py` | **The gate.** Builds the design matrix so nothing can see past noon |
| `models/base.py` | The shared contract every model implements |
| `models/baselines.py` | Persistence, climatology, ridge, Prophet |
| `models/gbm.py` | Boosted trees, LightGBM and XGBoost, point and quantile |
| `models/neural.py` | LSTM with a monotone quantile head |
| `models/conformal.py` | Split CQR and adaptive conformal inference |
| `evaluation/metrics.py` | MAE, CRPS, pinball, PICP, PINAW, Winkler |
| `evaluation/backtest.py` | Rolling-origin folds with a held-out calibration slice |
| `decision/dispatch.py` | The two-stage stochastic linear program |
| `decision/policies.py` | Oracle, point, quantile, stochastic, CVaR commitments |
| `drift/detectors.py` | ADWIN, Page-Hinkley, PSI, KS, on a standardised stream |
| `drift/policies.py` | Static, periodic and drift-triggered retraining |
| `explain/` | TreeSHAP attribution and error decomposition |
| `transfer.py` | Local, zero-shot, fine-tune and pooled strategies |
| `viz/plots.py` | Every figure in the report |

## If you only read three files

1. `src/riskaware_pv/data/features.py` for the gate, which everything depends on.
2. `src/riskaware_pv/decision/dispatch.py` for the optimiser that turns a forecast into money.
3. `tests/test_no_leakage.py` for the test that keeps the first one honest.

## Running it

```bash
make setup     # virtual environment and dependencies
make data      # download and cache the dataset
make test      # 22 tests
make e1        # ... through make e5, one per research question
make all       # everything, about 45 minutes
```
