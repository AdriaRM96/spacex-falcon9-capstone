"""Train the deployed prediction model and export it as a single artifact.

Run this at build time (see render.yaml) rather than committing a trained
model binary to the repo: training takes seconds on this dataset size, so
there's no real cost to regenerating it, and keeping "the model" as code
means it can't silently drift out of sync with the notebook's logic the way
a committed binary could.

Uses the original 90-launch dataset (dataset_part_3.csv), not the extended
672-launch one from notebook 08: retraining SVM on the extended data (see
notebook 07's retraining section) produced a degenerate model that predicts
the majority class for every input, because of severe class imbalance
(93.9% success). That's not something worth deploying behind a "predict my
launch" endpoint -- the original, smaller dataset's calibrated SVM is an
honest, if less data-rich, model.

Usage:
    python -m spacex_capstone.train_and_export [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from spacex_capstone.inference import build_baseline

DATASET_PART_3_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv"
)
DATASET_PART_2_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
)

# Same grid as notebooks/07_machine_learning_prediction.ipynb's SVM tuning
# cell, so the deployed model matches what the notebook actually validated.
SVM_PARAMETERS = {
    "kernel": ("linear", "rbf", "poly", "rbf", "sigmoid"),
    "C": np.logspace(-3, 3, 5),
    "gamma": np.logspace(-3, 3, 5),
}


def train_and_export(output_path: str | Path = "dashboard/model_artifact.joblib") -> None:
    data = pd.read_csv(DATASET_PART_2_URL)
    X = pd.read_csv(DATASET_PART_3_URL)
    Y = data["Class"].to_numpy()

    feature_names = X.columns.tolist()
    baseline = build_baseline(X)

    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC()
    svm_cv = GridSearchCV(svm, SVM_PARAMETERS, cv=10)
    svm_cv.fit(X_scaled, Y)
    print(f"Best SVM params: {svm_cv.best_params_}, CV accuracy: {svm_cv.best_score_:.4f}")

    # SVC() isn't fit with probability=True, so it has no predict_proba --
    # wrap it the same way notebook 07's business-impact section does.
    model = CalibratedClassifierCV(svm_cv.best_estimator_, cv=5)
    model.fit(X_scaled, Y)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "baseline": baseline,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"Wrote model artifact to {output_path}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "dashboard/model_artifact.joblib"
    train_and_export(output)
