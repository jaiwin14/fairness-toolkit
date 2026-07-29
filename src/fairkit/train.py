"""
Model training utilities.

`train_models` trains a set of standard classifiers on the same data with a
consistent interface, so downstream evaluation/mitigation code never needs
to know which library a given model came from.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# All model names this module knows how to build.
AVAILABLE_MODELS = ("logreg", "svc", "gbc", "xgboost")


def _build_estimators(random_state: int) -> dict:
    """Construct fresh, unfitted estimator instances."""
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        # probability=True is required so evaluate.py / mitigate.py can use
        # predict_proba (needed by ROC and Calibrated EqOdds mitigation).
        "svc": SVC(probability=True, random_state=random_state),
        "gbc": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state, eval_metric="logloss"),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: tuple[str, ...] = AVAILABLE_MODELS,
    random_state: int = 42,
) -> dict[str, object]:
    """
    Fit each requested model on (X_train, y_train).

    Parameters
    ----------
    X_train, y_train : training features and labels.
    models : which models to train, by name (subset of AVAILABLE_MODELS).
    random_state : shared random seed for reproducibility across models.

    Returns
    -------
    dict[str, object]
        Mapping of model name -> fitted estimator.
    """
    estimators = _build_estimators(random_state)

    unknown = set(models) - set(estimators)
    if unknown:
        raise ValueError(f"Unknown model(s) {sorted(unknown)}. Available: {list(estimators)}")

    fitted = {}
    for name in models:
        model = estimators[name]
        model.fit(X_train, y_train)
        fitted[name] = model

    return fitted
