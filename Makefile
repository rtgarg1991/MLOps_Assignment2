PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
FLAKE8 ?= .venv/bin/flake8
UVICORN ?= .venv/bin/uvicorn

.PHONY: install lint test quality ingest preprocess train serve smoke eval-batch submission pipeline

pipeline: quality ingest preprocess train

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(FLAKE8) src tests

test:
	PYTHONPATH=. $(PYTEST) -q

quality: lint test

ingest:
	PYTHONPATH=. $(PYTHON) src/ingest.py --source-dir data/raw --output-dir data/ingested

preprocess:
	PYTHONPATH=. $(PYTHON) src/preprocessing.py --input-dir data/ingested --output-dir data/processed --image-size 224 --train-ratio 0.8 --val-ratio 0.1 --seed 42

train:
	PYTHONPATH=. $(PYTHON) src/train.py --data-dir data/processed --output-model models/model.pt --artifact-dir artifacts/training --epochs 5 --batch-size 32 --learning-rate 0.001 --model-variant baseline --num-workers 0 --force-cpu

serve:
	PYTHONPATH=. ENABLE_REAL_MODEL=1 MODEL_PATH=$(PWD)/models/model.pt $(UVICORN) src.main:app --host 0.0.0.0 --port 8080

smoke:
	PYTHONPATH=. $(PYTHON) src/smoke_test.py --base-url http://127.0.0.1:8080

eval-batch:
	PYTHONPATH=. $(PYTHON) scripts/create_eval_batch.py --processed-dir data/processed --output-csv artifacts/eval_batch.csv --per-class 20 --seed 42

submission:
	./scripts/create_submission_bundle.sh
