# Assignment 2 Coverage Matrix

## M1: Model Development and Experiment Tracking
1. Data and code versioning:
- Code versioning via Git repository.
- DVC pipeline and config in `dvc.yaml`, `.dvc/config`, `params.yaml`.

2. Model building:
- Baseline and wider CNN options implemented in `src/model.py`.
- Training pipeline in `src/train.py`.
- Serialized model artifact as `.pt`.

3. Experiment tracking:
- MLflow integration in `src/train.py` (params, metrics, artifacts).
- Artifacts include confusion matrix and loss curve.

## M2: Model Packaging and Containerization
1. Inference service:
- FastAPI service in `src/main.py`.
- Endpoints:
  - `GET /health`
  - `POST /predict`

2. Environment specification:
- Pinned dependencies in `requirements.txt`.

3. Containerization:
- Training image: `Dockerfile`.
- Serving image: `Dockerfile.serve`.

## M3: CI Pipeline for Build, Test and Image Creation
1. Automated tests:
- Preprocessing utility tests: `tests/test_preprocessing.py`.
- Inference utility tests: `tests/test_inference_utils.py`.
- API tests: `tests/test_api.py`.

2. CI setup:
- GitHub Actions workflow: `.github/workflows/mlops-pipeline.yml`.
- Includes install, lint, tests, container build.

3. Artifact publishing:
- Workflow pushes trainer and serving images to Artifact Registry on `push`.

## M4: CD Pipeline and Deployment
1. Deployment target:
- Kubernetes manifests under `k8s/`.
- Main deployment manifest: `k8s/deployment.yaml`.

2. CD and GitOps flow:
- On `push` to `main`: deploy to GKE with new serving image.

3. Smoke tests and health checks:
- Post-deploy smoke test script: `src/smoke_test.py`.
- Workflow fails if smoke test fails.

## M5: Monitoring, Logging and Final Submission
1. Basic monitoring and logging:
- API request logging middleware in `src/main.py`.
- Metrics via Prometheus counters and histograms in `src/main.py`.
- GKE PodMonitoring manifest: `k8s/monitoring/google-pod-monitor.yaml`.

2. Post-deployment model performance tracking:
- Evaluation script with true labels: `src/post_deploy_eval.py`.
- Eval batch generator: `scripts/create_eval_batch.py`.

3. Submission packaging:
- Bundle script: `scripts/create_submission_bundle.sh`.
- Includes all required source/config/manifests/docs assets.

## Local Validation Evidence
1. Lint and tests:
- `flake8` pass.
- `pytest` pass.

2. Pipeline on real dataset path:
- Real data integrity checks completed.
- Ingest, preprocess, EDA, train, API, smoke checks executed.

3. Runtime artifacts:
- Model artifact (`.pt`), metrics JSON, confusion matrix PNG, loss curve PNG generated.
