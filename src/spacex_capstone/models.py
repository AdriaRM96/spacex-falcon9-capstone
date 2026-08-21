"""Model tuning and evaluation utilities.

Backs notebook 07: hyperparameter search, the confusion-matrix plot used
throughout, and a robust cross-validated evaluation that goes beyond a
single train/test split.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate


def plot_confusion_matrix(y: np.ndarray, y_predict: np.ndarray) -> None:
    """Plot a confusion matrix heatmap for true vs. predicted labels."""
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["did not land", "landed"])
    ax.yaxis.set_ticklabels(["did not land", "landed"])
    plt.show()


def tune_model(
    estimator: BaseEstimator,
    param_grid: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 10,
) -> GridSearchCV:
    """Fit a `GridSearchCV` over `param_grid` and return the fitted search object.

    Args:
        estimator: An unfitted scikit-learn estimator.
        param_grid: Hyperparameter grid, as passed to `GridSearchCV`.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of cross-validation folds used during the search.

    Returns:
        The fitted `GridSearchCV` object (`.best_estimator_`, `.best_params_`,
        and `.best_score_` are available on it).
    """
    search = GridSearchCV(estimator, param_grid, cv=cv)
    search.fit(X_train, y_train)
    return search


def evaluate_with_stratified_kfold(
    models: dict[str, BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 2,
    scoring: tuple[str, ...] = ("accuracy", "precision", "recall", "f1"),
) -> pd.DataFrame:
    """Compare multiple models with `StratifiedKFold` cross-validation.

    Reports each metric as mean and standard deviation across folds, which
    is far more informative than a single accuracy number from one
    train/test split -- especially on small datasets where one split can be
    unrepresentative.

    Models are cloned and, where supported, given a fixed `random_state`
    before evaluation, so results are reproducible across runs even for
    models with stochastic components (e.g. a Decision Tree with
    `splitter='random'`).

    Args:
        models: Mapping of model name to a fitted or unfitted estimator
            (only its hyperparameters are used; `cross_validate` refits on
            each fold internally).
        X: Full feature matrix (already preprocessed/standardized).
        y: Full label vector.
        n_splits: Number of stratified folds.
        random_state: Seed for both the fold split and any model that
            accepts a `random_state` parameter.
        scoring: Metric names passed to `cross_validate`.

    Returns:
        A DataFrame indexed by model name, with `{metric}_mean` and
        `{metric}_std` columns for each metric in `scoring`.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for name, model in models.items():
        model = clone(model)
        if "random_state" in model.get_params():
            model.set_params(random_state=random_state)

        scores = cross_validate(model, X, y, cv=skf, scoring=list(scoring))
        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"{metric}_mean"] = scores[key].mean()
            row[f"{metric}_std"] = scores[key].std()
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")
