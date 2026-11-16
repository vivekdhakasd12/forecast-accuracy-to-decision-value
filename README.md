# From Forecast Accuracy to Decision Value

**Decision-focused learning for day-ahead renewable energy commitment.**
Master's thesis, M.Sc. Data Science and Artificial Intelligence, SRH University of Applied Sciences Heidelberg.

A solar farm has to promise the grid, by noon, how much power it will deliver in every hour
of tomorrow. Being wrong costs money. Forecasters are normally trained to be *accurate*;
this thesis trains them on the *euro cost of the decisions they lead to*, and measures when
that pays off, by how much, and what it costs in honesty about uncertainty.

New to the topic? Read [thesis/docs/THESIS_IN_PLAIN_WORDS.md](thesis/docs/THESIS_IN_PLAIN_WORDS.md) first.

---

## What is in this folder

```
forecast-accuracy-to-decision-value/
├── README.md              you are here
├── .vscode/               one-click tasks for VS Code (see below)
├── src/riskaware_pv/      all the code, shared by the case study and the thesis
│   ├── data/              loaders: OPSD (case study), SMARD + reBAP (thesis)
│   ├── models/            forecasting models and conformal calibration
│   ├── decision/          the dispatch optimiser that turns forecasts into euros
│   ├── drift/  explain/  evaluation/  viz/
├── thesis/
│   ├── docs/              exposé (signed version), review, plain-words plan
│   ├── experiments/       thesis experiments (added step by step)
│   └── notebooks/         demo notebooks
├── case_study_1/          the finished case study this thesis builds on
│   ├── README.md          what it found and how to reproduce it
│   ├── experiments/       e1..e5
│   ├── results/ figures/  outputs
│   ├── report/            48-page report (LaTeX + PDF)
│   └── presentation/      slides, demo notebook, videos
├── tests/                 41 tests: leakage, dispatch economics, data loaders
├── configs/default.yaml   every tunable number
├── data/                  downloaded data (not in git, rebuilt with make)
├── .env.example           template for the private API keys
└── Makefile               all commands
```

## Running it in VS Code

1. **Open the folder**: File, Open Folder, choose `forecast-accuracy-to-decision-value`.
2. **Install the recommended extensions** when VS Code asks (Python, Jupyter).
3. **Run a task**: press `Cmd+Shift+P`, type *Run Task*, and pick one:

| Task | What it does | Time |
|---|---|---|
| **Setup: install Python environment** | creates `.venv` with everything | ~3 min, once |
| **Tests: run all** | the 41 tests | ~10 s |
| **Thesis: build German market data** | SMARD + reBAP, 2019 to Oct 2026 | ~1 min (cached) |
| **Case Study 1: run experiment e1 … e5** | reproduces a case study result | 2 to 15 min each |

Tests also appear in the Testing panel (the flask icon on the left).

The same things from a terminal:

```bash
make setup      # once
make test
make data-de    # thesis data
make help       # everything else
```

## Data

| What | Source | Period | Notes |
|---|---|---|---|
| German solar output, TSO solar forecast, load, day-ahead price | [SMARD](https://www.smard.de) (Bundesnetzagentur), CC BY 4.0 | Jan 2019 to Oct 2026, hourly | free, no account |
| Installed solar capacity | [Energy-Charts](https://www.energy-charts.info) (Fraunhofer ISE) | monthly | free, no account |
| Imbalance price (reBAP) | [Netztransparenz](https://www.netztransparenz.de) (German TSOs) | Jan 2019 to Oct 2026, 15-min | free account; keys go in `.env` |
| Case Study 1 and other countries | [Open Power System Data](https://data.open-power-system-data.org/time_series/) | 2015 to Sep 2020 | used for transfer experiments |

To get the reBAP keys, copy `.env.example` to `.env` and follow the note inside. `.env` is
never committed.

## Status

| Weeks | Work | State |
|---|---|---|
| 7-8 | Real German data, dispatch model safe for real prices | data done; timing and commitment-cost fixes next |
| 9 | Baselines on the new data | |
| 10-11 | Decision-focused method 1: cost-weighted quantile loss | |
| 12-13 | Method 2: SPO+ (adapted) | |
| 14-15 | Method 3: differentiable optimisation layer | |
| 16 | Calibration audit + conformal on top | |
| 17 | Drift and transfer | |
| 18 | Penalty x storage grid, statistics | |
| 20-24 | Writing | |

The full plan is in [thesis/docs/THESIS_IN_PLAIN_WORDS.md](thesis/docs/THESIS_IN_PLAIN_WORDS.md),
the detailed review of the exposé in [thesis/docs/expose_review.md](thesis/docs/expose_review.md).
