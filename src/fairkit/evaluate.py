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


def disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    privileged_value: int = 0,
    favorable_label: int = 0,
) -> float:
    """
    Ratio of favorable-outcome rates: unprivileged group / privileged group.

    A ratio of 1.0 is the fairness ideal (both groups equally likely to
    receive the favorable prediction). Values far from 1.0 in either
    direction indicate disparate treatment.

    `favorable_label` (default 0) matches `fairkit.datasets.compas`'s
    convention; pass the value from whichever dataset module you're using
    (e.g. `fairkit.datasets.adult.FAVORABLE_LABEL`, which is 1) — this
    function makes no assumption about which class is "good."

    `privileged_value` is which value of the sensitive column counts as
    "privileged" (default 0, matching `fairkit.mitigate`'s convention for
    one-hot indicator columns like `race_African-American` or `sex_Male`,
    where 0 means "not a member of the named group").
    """
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)

    unprivileged_mask = sensitive_features != privileged_value
    privileged_mask = sensitive_features == privileged_value

    if not unprivileged_mask.any() or not privileged_mask.any():
        return float("nan")

    unpriv_rate = np.mean(y_pred[unprivileged_mask] == favorable_label)
    priv_rate = np.mean(y_pred[privileged_mask] == favorable_label)

    if priv_rate == 0:
        return float("inf") if unpriv_rate > 0 else float("nan")
    return unpriv_rate / priv_rate


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_col: str,
    y_pred: np.ndarray | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    mitigation: str | None = None,
    favorable_label: int = 0,
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
    favorable_label : which label value is the favorable outcome, for the
        disparate impact ratio (default 0, COMPAS's convention — pass the
        dataset module's own FAVORABLE_LABEL for other datasets, e.g.
        `fairkit.datasets.adult.FAVORABLE_LABEL`).

    Returns
    -------
    dict with keys: model_name, dataset_name, mitigation, sensitive_col,
        accuracy, demographic_parity_difference, equalized_odds_difference,
        disparate_impact_ratio, accuracy_by_group.
    """
    if y_pred is None:
        if model is None:
            raise ValueError("Must supply either a fitted model or precomputed y_pred.")
        y_pred = model.predict(X_test)

    sensitive_features = X_test[sensitive_col]

    accuracy = accuracy_score(y_test, y_pred)
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
    di_ratio = disparate_impact_ratio(y_pred, sensitive_features, favorable_label=favorable_label)

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
        "disparate_impact_ratio": di_ratio,
        "accuracy_by_group": by_group,
    }
