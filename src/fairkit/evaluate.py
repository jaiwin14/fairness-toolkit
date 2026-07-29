"""
Evaluation utilities: accuracy + fairness metrics for a fitted model.

`evaluate_model` is the single function downstream benchmark scripts call.
It returns a flat dict (easy to append into a results DataFrame/CSV row)
combining overall accuracy with Fairlearn's standard group-fairness metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_col: str,
    y_pred: np.ndarray | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    mitigation: str | None = None,
) -> dict:
    """
    Evaluate a fitted model's accuracy and fairness on held-out data.

    Parameters
    ----------
    model : a fitted estimator with .predict(X) (ignored if y_pred is given;
        pass model=None when scoring externally-produced/mitigated predictions).
    X_test, y_test : held-out features and true labels.
    sensitive_col : column name in X_test to use as the protected attribute.
    y_pred : optional precomputed predictions (e.g. from a mitigation
        technique). If omitted, `model.predict(X_test)` is used.
    model_name, dataset_name, mitigation : optional labels carried through
        into the output row, for easy identification in a results table.

    Returns
    -------
    dict with keys: model_name, dataset_name, mitigation, sensitive_col,
        accuracy, demographic_parity_difference, equalized_odds_difference,
        accuracy_by_group.
    """
    if y_pred is None:
        if model is None:
            raise ValueError("Must supply either a fitted model or precomputed y_pred.")
        y_pred = model.predict(X_test)

    sensitive_features = X_test[sensitive_col]

    accuracy = accuracy_score(y_test, y_pred)
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)

    by_group = MetricFrame(
        metrics={"accuracy": accuracy_score},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    ).by_group["accuracy"].to_dict()

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "mitigation": mitigation,
        "sensitive_col": sensitive_col,
        "accuracy": accuracy,
        "demographic_parity_difference": dp_diff,
        "equalized_odds_difference": eo_diff,
        "accuracy_by_group": by_group,
    }
