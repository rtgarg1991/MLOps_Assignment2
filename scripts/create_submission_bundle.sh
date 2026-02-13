#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="submission"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="mlops_assignment2_bundle_${STAMP}.zip"

mkdir -p "${OUT_DIR}"

zip -r "${OUT_DIR}/${BUNDLE_NAME}" \
  src tests k8s .github \
  Dockerfile Dockerfile.serve requirements.txt \
  README.md dvc.yaml params.yaml .dvc/config \
  docs scripts \
  models artifacts \
  -x "*/__pycache__/*" \
     "*.pyc" \
     ".venv/*" \
     ".git/*" \
     "reference_from_assignment1/*" \
     "local_test/*" \
     "local_test2/*" \
     "local_actual_validation/*" \
     "mlruns/*"

printf 'Created submission bundle: %s\n' "${OUT_DIR}/${BUNDLE_NAME}"
