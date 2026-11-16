PY := .venv/bin/python
CS1 := case_study_1/experiments
export PYTHONPATH := src

.PHONY: help setup test data data-de cs1-e1 cs1-e2 cs1-e3 cs1-e4 cs1-e5 cs1-all clean-cs1 \
        e1 e2 e3 e4 e5 all

help:
	@echo "Setup"
	@echo "  make setup     create .venv and install everything"
	@echo "  make test      run all tests (leakage, dispatch economics, data loaders)"
	@echo ""
	@echo "Thesis"
	@echo "  make data-de   German solar, load, day-ahead price (SMARD) + reBAP imbalance price"
	@echo "                 (reBAP needs your keys in .env, see .env.example)"
	@echo ""
	@echo "Case Study 1 (finished, reproducible)"
	@echo "  make data      download OPSD and build the design matrix"
	@echo "  make cs1-e1    probabilistic forecasting"
	@echo "  make cs1-e2    risk-aware dispatch"
	@echo "  make cs1-e3    concept drift and retraining"
	@echo "  make cs1-e4    explainability"
	@echo "  make cs1-e5    transfer to a new site (small data)"
	@echo "  make cs1-all   tests + all five experiments (~45 min)"

setup:
	uv venv --python 3.11 .venv || python3 -m venv .venv
	uv pip install --python $(PY) -e ".[deep,dev,notebook]" || $(PY) -m pip install -e ".[deep,dev,notebook]"
	@echo "macOS note: LightGBM/XGBoost need OpenMP -> brew install libomp"

test:
	$(PY) -m pytest tests -q

# ---- thesis ----------------------------------------------------------------
data-de:
	$(PY) -m riskaware_pv.data.smard --force

# ---- Case Study 1 ------------------------------------------------------------
data:
	$(PY) -c "from riskaware_pv.pipeline import load_design_matrix; \
	from riskaware_pv.config import Config; \
	X = load_design_matrix(Config.load(), force=True); print(X.shape)"

cs1-e1:
	$(PY) -W ignore $(CS1)/e1_forecasting.py
cs1-e2:
	$(PY) -W ignore $(CS1)/e2_dispatch.py --max-days 200 --all-models
cs1-e3:
	$(PY) -W ignore $(CS1)/e3_drift.py
cs1-e4:
	$(PY) -W ignore $(CS1)/e4_explain.py
cs1-e5:
	$(PY) -W ignore $(CS1)/e5_transfer.py

cs1-all: test cs1-e1 cs1-e2 cs1-e3 cs1-e4 cs1-e5
	@echo "Case Study 1 complete -- see case_study_1/results/ and case_study_1/figures/"

clean-cs1:
	rm -rf case_study_1/results/* case_study_1/figures/* data/processed/* data/interim/opsd_*

# old names still work
e1: cs1-e1
e2: cs1-e2
e3: cs1-e3
e4: cs1-e4
e5: cs1-e5
all: cs1-all
