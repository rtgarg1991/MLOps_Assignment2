# MLOps Assignment 2 - Video Presentation Script

## Prep Checklist Before Recording
- [ ] Open the project report document showing Group Name, Member Names, and IDs.
- [ ] Have the GitHub repository open on the `main` branch.
- [ ] Open the GitHub Actions "Actions" tab.
- [ ] Open Google Cloud Console with the GCS Bucket open (`datasets/cats-dogs`).
- [ ] Open the MLflow UI in the browser.
- [ ] Open the GCP Cloud Monitoring Dashboard ("Cats vs Dogs API Dashboard").
- [ ] Have Postman open with the `Cats vs Dogs API` collection imported.
- [ ] Open VS Code / IDE with the project codebase showing `src/train.py`, `.github/workflows/mlops-pipeline.yml`, and `dvc.yaml`.

---

## 🕒 [00:00 - 01:00] Part 1: Introduction & Project Overview

**Visual:** 
- Start recording on the front page of your project report document.
- Highlight the group name and member details one by one.

**Speaker (You/Team Member):**
*"Hello everyone! Welcome to our MLOps Assignment 2 presentation. We are Group [Insert Group Name/Number], and our team consists of five members: [Read Names and IDs briefly]."**

**Visual:** 
- Switch to the GitHub repository homepage or VS Code showing `README.md`.

**Speaker:**
*"For this assignment, we built an end-to-end MLOps pipeline for an image classification problem—specifically, classifying Cats vs. Dogs using PyTorch. Our tech stack includes:*
- *GitHub Actions for CI/CD.*
- *Google Cloud Storage (GCS) for data versioning and model storage.*
- *DVC for local data versioning.*
- *Google Kubernetes Engine (GKE) for both training jobs and API deployment.*
- *MLflow for experiment tracking.*
- *FastAPI for model serving.*
- *And Prometheus with GCP Cloud Monitoring for observability."*

---

## 🕒 [01:00 - 02:30] Part 2: Data Versioning & the Pipeline

**Visual:** 
- Open VS Code and show `dvc.yaml` and `k8s/production-train-job.yaml`.
- Switch to Google Cloud Console > Cloud Storage and show the bucket structure (e.g., `gs://<bucket-name>/datasets/cats-dogs/<git-sha>/`).

**Speaker:**
*"Let’s walk through our data pipeline. All data preprocessing is fully automated and version-controlled. We use DVC locally (`dvc.yaml`) to track data changes across our `ingest`, `preprocess`, `eda` and `train` stages.*

*In production, when a PR is merged, our GitHub Actions pipeline triggers a Kubernetes Job. This job downloads raw data from GCS, runs ingestion, splits the data, performs Exploratory Data Analysis, and then uploads the versioned artifacts back to GCS.*

*Here in the GCS bucket, you can see how data is organized tightly by Git commit SHAs. For example, if we look inside this recent commit folder, we have subdirectories for `ingested`, `processed`, and `eda` outputs, like our `class_balance.png` chart. This ensures perfect data-to-code lineage for every single training run."*

---

## 🕒 [02:30 - 04:00] Part 3: Model Training & MLflow

**Visual:** 
- Show `src/train.py` in VS Code briefly to highlight MLflow auto-logging or custom logging.
- Switch to the MLflow UI in the browser.

**Speaker:**
*"Once data is ready, the K8s job trains a PyTorch CNN model. We utilize MLflow, which we deployed on our GKE cluster, for comprehensive experiment tracking."*

*In the MLflow UI here, you can see our past runs. For each run, MLflow tracks hyperparameters like `epochs`, `batch_size`, and `learning_rate`.*

- **Action:** Click on one of the recent successful runs.

**Speaker:**
*"Inside a run, we log validation metrics like `test_accuracy` and `test_loss`. We also log essential artifacts: the trained PyTorch `model.pt`, a confusion matrix, and a loss curve. This central tracking makes it trivial to compare models over time and select the best one for deployment."*

---

## 🕒 [04:00 - 05:30] Part 4: CI/CD Pipeline (GitHub Actions)

**Visual:** 
- Switch to GitHub > Actions tab.
- Open a recent successful run of `MLOps Assignment 2 Pipeline`.

**Speaker:**
*"Our CI/CD is orchestrated by GitHub Actions. The pipeline consists of several interconnected jobs:*
1. *`lint-and-test`: First, we enforce code quality with `flake8` and run unit tests with `pytest`. This runs on both PRs and pushes to main.*
2. *`build-trainer-image`: If tests pass, we build the training Docker image and push it to Google Artifact Registry.*
3. *`run-training-on-gke`: We submit the training job to K8s. The pipeline polls K8s, handles credential refreshing, and waits for training completion.*
4. *`build-and-push-api-image`: It downloads the newly trained model from GCS, bakes it into the serving API image, and pushes it.*
5. *`deploy-to-gke`: Finally, the new API image is deployed to the GKE cluster."*

---

## 🕒 [05:30 - 06:30] Part 5: Deployment & API Testing

**Visual:** 
- Open Postman with the "Cats vs Dogs API" collection loaded.
- Show the runner if possible, or just the individual requests.

**Speaker:**
*"The production model sits behind a FastAPI service exposed via a Kubernetes LoadBalancer. To ensure reliability, the CI/CD pipeline runs a built-in `smoke_test.py` immediately after deployment. Let me showcase the API manually using Postman.*

- **Action:** Send a `GET /health` request.
*"Our `/health` endpoint confirms the service is up and whether the real PyTorch model was successfully loaded."*

- **Action:** Send a `POST /predict` request holding `Cat/0.jpg`.
*"Let's send a cat image to the `/predict` endpoint. As you can see, the model correctly responds with the label and confidence probabilities."*

---

## 🕒 [06:30 - 08:00] Part 6: Monitoring & Observability

**Visual:** 
- Switch to the GCP Cloud Monitoring Dashboard in the browser.

**Speaker:**
*"A critical pillar of MLOps is monitoring. We configured our FastAPI application with `prometheus_client` to expose custom metrics on the `/metrics` endpoint.*

*Using Google Cloud Managed Service for Prometheus, metrics are scraped automatically via a PodMonitoring resource in K8s. Here is our GCP Metrics Explorer Dashboard:*

- **Action:** Briefly hover over the distinct panels.

*"We constructed panels to monitor both system and ML health in real-time:*
- *Traffic and Latency: Request rates, P50/P99 latencies, and 4xx/5xx error rates.*
- *ML Metrics: Prediction label ratios, inference latency, and crucially, statistical confidence distributions.*
- *Data quality: Upload sizes and distinct prediction prediction errors (like invalid image types).*
- *Infrastructure: Memory, CPU, and in-flight requests.*

*This robust observability enables us to immediately detect data drift or infrastructure bottlenecks."*

---

## 🕒 [08:00 - 08:30] Part 7: Conclusion

**Visual:** 
- Show a final architecture slide if you have one, or just the GitHub repo homepage.

**Speaker:**
*"To conclude, our project bridges the gap between machine learning code and a practical, robust, and scalable production system. By heavily utilizing GitOps, containerization, remote data versioning, and rigorous ML tracking with MLflow and Prometheus, we established a completely reproducible and observable ML operation.*

*Thank you for watching!"*
