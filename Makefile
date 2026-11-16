PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: help setup data data-de test e1 e2 e3 e4 e5 all clean

help:
	@echo "make setup   - create .venv and install dependencies"
	@echo "make data    - download OPSD and build the design matrix"
	@echo "make data-de - thesis data: German PV, load, day-ahead price (SMARD) + reBAP if available"
	@echo "make test    - run the test suite (leakage + dispatch properties)"
	@echo "make e1      - experiment 1: probabilistic forecasting"
	@echo "make e2      - experiment 2: risk-aware dispatch"
	@echo "make e3      - experiment 3: concept drift and retraining"
	@echo "make e4      - experiment 4: explainability"
	@echo "make e5      - experiment 5: transfer to a new site (small data)"
	@echo "make all     - test + e1..e4 end to end"

setup:
	uv venv --python 3.11 .venv || python3 -m venv .venv
	uv pip install --python $(PY) -e ".[deep,dev]" || $(PY) -m pip install -e ".[deep,dev]"
	@echo "macOS note: LightGBM/XGBoost need OpenMP -> brew install libomp"

data:
	$(PY) -c "from riskaware_pv.pipeline import load_design_matrix; \
	from riskaware_pv.config import Config; \
	X = load_design_matrix(Config.load(), force=True); print(X.shape)"

data-de:
	$(PY) -m riskaware_pv.data.smard --force

test:
	$(PY) -m pytest tests -q

e1:
	$(PY) -W ignore experiments/e1_forecasting.py
e2:
	$(PY) -W ignore experiments/e2_dispatch.py --max-days 200 --all-models
e3:
	$(PY) -W ignore experiments/e3_drift.py
e4:
	$(PY) -W ignore experiments/e4_explain.py
e5:
	$(PY) -W ignore experiments/e5_transfer.py

all: test e1 e2 e3 e4 e5
	@echo "All experiments complete -- see results/ and figures/"

clean:
	rm -rf results/* figures/* data/processed/* data/interim/*
