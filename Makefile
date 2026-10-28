# Reproduce everything:  make data train analytics notebook test
PY ?= python
export PYTHONPATH := src:api

data:        ## Phase 1: clean + engineer features (needs data/raw/ctg-studies.csv)
	$(PY) -m trialsentinel.data
train:       ## Phase 3: tune, select, calibrate, explain, export (~40 min on 2 cores)
	$(PY) -m trialsentinel.train
train-quick: ## Smoke test with a handful of Optuna trials
	$(PY) -m trialsentinel.train --quick --timeout 120
onnx:        ## Export the selected model to ONNX and verify parity
	$(PY) scripts/export_onnx.py
analytics:   ## Build overview.json for the analytics endpoint and dashboard snapshot
	$(PY) -m trialsentinel.analytics && cp artifacts/overview.json web/data/overview.json
notebook:    ## Phase 2: regenerate and execute the EDA + model notebook
	$(PY) scripts/build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/trialsentinel_eda_and_model.ipynb
api:         ## Phase 4: run the API locally on :8000
	cd api && PYTHONPATH=../src uvicorn app.main:app --reload --port 8000
test:
	$(PY) -m pytest api/tests -q
web:         ## Phase 5: run the dashboard on :3000 (needs the API running)
	cd web && npm install && API_URL=http://localhost:8000 npm run dev
all: data train onnx analytics notebook test

.PHONY: data train train-quick onnx analytics notebook api test web all
