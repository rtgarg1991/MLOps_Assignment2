# MLOps Assignment 2: Cats vs Dogs Classification (GKE)

This repository implements an end-to-end MLOps pipeline for binary image classification (cats vs dogs), aligned to Assignment 2 requirements (M1-M5).

## Scope

- Binary image classification API for a pet adoption use case
- Dataset preprocessing to `224x224` RGB
- Train/val/test split (default `80/10/10`)
- Baseline CNN training with reproducible parameters
- Experiment logging support (MLflow integration in training script)
- Containerized inference API (`FastAPI`)
- CI/CD workflow for image build, test, deploy, and smoke checks
- GKE deployment manifests and monitoring hooks
- Post-deployment labeled-batch evaluation utility
- DVC pipeline definitions for reproducibility

## Project structure

- `src/ingest.py`: Ingest and normalize raw cat/dog image folders
- `src/preprocessing.py`: Resize and split dataset
- `src/train.py`: Model training and artifact generation
- `src/model.py`: CNN model definitions
- `src/main.py`: FastAPI inference service (`/health`, `/predict`, `/metrics`)
- `src/smoke_test.py`: API smoke test script
- `src/post_deploy_eval.py`: Post-deploy evaluation using labeled CSV
- `scripts/create_eval_batch.py`: Build labeled eval CSV from test split
- `scripts/create_submission_bundle.sh`: Build submission zip
- `tests/`: Unit/API test suite
- `k8s/`: Kubernetes deployment and job manifests
- `.github/workflows/mlops-pipeline.yml`: CI/CD pipeline
- `dvc.yaml`, `params.yaml`, `.dvc/config`: DVC pipeline setup
- `docs/ASSIGNMENT2_REQUIREMENTS_COVERAGE.md`: Requirement-to-implementation matrix

## Prerequisites

- macOS/Linux shell
- Python 3.11+ (3.12 tested)
- Docker Desktop
- (For cloud deploy) `gcloud`, GKE access, Artifact Registry, Workload Identity setup

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset expected layout

The pipeline expects raw images under:

- `data/raw/cat/*.jpg`
- `data/raw/dog/*.jpg`

Case-insensitive source folders are handled by ingest logic (`Cat`/`Dog` also work).

## Local training pipeline

### 1. Ingest raw dataset

```bash
PYTHONPATH=. .venv/bin/python src/ingest.py \
  --source-dir data/raw \
  --output-dir data/ingested
```

This normalizes folder names (`Cat/Dog` → `cat/dog`) so the pipeline is case-insensitive and works on both macOS and Linux.

### 2. Preprocess and split

```bash
PYTHONPATH=. .venv/bin/python src/preprocessing.py \
  --input-dir data/ingested \
  --output-dir data/processed \
  --image-size 224 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

### 3. Train baseline model

```bash
PYTHONPATH=. .venv/bin/python src/train.py \
  --data-dir data/processed \
  --output-model models/model.pt \
  --artifact-dir artifacts/training \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --model-variant baseline \
  --num-workers 0 \
  --force-cpu
```

Generated outputs:

- `models/model.pt`
- `artifacts/training/metrics.json`
- `artifacts/training/confusion_matrix.png`
- `artifacts/training/loss_curve.png`

## Run API locally

### Start API (real model)

```bash
PYTHONPATH=. ENABLE_REAL_MODEL=1 MODEL_PATH=$(pwd)/models/model.pt \
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080
```

### Health check

```bash
curl http://127.0.0.1:8080/health
```

### Prediction

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
  -F "file=@/absolute/path/to/sample.jpg"
```

## Docker validation

### Build serving image

```bash
docker build -f Dockerfile.serve -t cats-dogs-api-local:latest .
```

### Run container

```bash
docker run --rm -p 8080:8080 \
  -e ENABLE_REAL_MODEL=1 \
  -e MODEL_PATH=/app/models/model.pt \
  cats-dogs-api-local:latest
```

### Validate endpoints

```bash
curl http://127.0.0.1:8080/health
curl -X POST "http://127.0.0.1:8080/predict" -F "file=@/absolute/path/to/sample.jpg"
```

## Tests and lint

```bash
PYTHONPATH=. .venv/bin/pytest -q
.venv/bin/flake8 src tests scripts
```

## DVC workflow

DVC files are present and pipeline stages are defined.

Run:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=. dvc repro
```

If your local machine is CPU-only and dataset is large, training stage may take significant time.

## CI/CD overview

Workflow file:

- `.github/workflows/mlops-pipeline.yml`

Pipeline behavior:

1. Lint + tests
2. Build trainer image
3. (push only) Push trainer image
4. (push only) Run training job on GKE
5. Build and push API image
6. Deploy to GKE
7. Run smoke test gate
8. Optional post-deploy evaluation when `EVAL_BATCH_GCS_URI` is configured

## Required GitHub secrets (for cloud)

- `GCP_PROJECT_ID`
- `GCP_WIF_PROVIDER`
- `GCP_SA_EMAIL`
- `GCP_BUCKET_NAME`
- `GKE_CLUSTER_NAME`
- `GKE_CLUSTER_LOCATION`
- `ARTIFACT_REGISTRY_REPO`
- `EVAL_BATCH_GCS_URI` (optional)

## Kubernetes manifests

- `k8s/deployment.yaml`: API deployment + service
- `k8s/production-train-job.yaml`: full training job template
- `k8s/parallel-train-job.yaml`: variant comparison training
- `k8s/universal-job.yaml`: generic job wrapper
- `k8s/monitoring/google-pod-monitor.yaml`: metrics scraping

## Post-deployment model performance tracking

### Create eval batch CSV

```bash
PYTHONPATH=. .venv/bin/python scripts/create_eval_batch.py \
  --processed-dir data/processed \
  --output-csv artifacts/eval_batch.csv \
  --per-class 100 \
  --seed 42
```

### Evaluate deployed API

```bash
PYTHONPATH=. .venv/bin/python src/post_deploy_eval.py \
  --base-url http://<api-url> \
  --input-csv artifacts/eval_batch.csv \
  --output-json artifacts/post_deploy_eval.json
```

## Submission packaging

Create assignment bundle:

```bash
./scripts/create_submission_bundle.sh
```

Output zip is created under `submission/`.

## Assignment requirement coverage

See:

- `docs/ASSIGNMENT2_REQUIREMENTS_COVERAGE.md`

This maps M1-M5 requirements to concrete files and workflow steps.
