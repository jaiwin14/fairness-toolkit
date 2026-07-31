"""
Bias mitigation techniques, wrapping AIF360.

Two families of technique are wrapped here, and they have different shapes
because AIF360 treats them differently:

- POST-PROCESSING (`reject_option_classification`, `equalized_odds`,
  `calibrated_equalized_odds`): take an *already-fitted* model plus test
  data, and adjust its predictions after the fact. Signature:
  `fn(model, X_test, y_test, sensitive_col, ...) -> np.ndarray` of
  mitigated predictions on X_test.

- IN-PROCESSING (`adversarial_debiasing`): trains its own classifier from
  scratch with a fairness objective baked into training. It does not wrap
  an existing model. Signature:
  `fn(X_train, y_train, X_test, sensitive_col, ...) -> np.ndarray` of
  predictions on X_test.

All of them convert pandas data to AIF360's `BinaryLabelDataset`, which is
the fiddliest part of this module: AIF360 expects one combined DataFrame
(features + label column), explicit favorable/unfavorable label values, and
"privileged"/"unprivileged" groups expressed as dicts over the protected
attribute's *encoded* column (e.g. `{"race_African-American": 0}`).

Convention used throughout this module: label value 0 = favorable
(did not reoffend), label value 1 = unfavorable (reoffended) — matching
`is_recid` / `two_year_recid` as produced by the dataset loaders directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset


def _to_binary_label_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive_col: str,
    favorable_label: int = 0,
    unfavorable_label: int = 1,
) -> BinaryLabelDataset:
    """Pack features + labels into the BinaryLabelDataset AIF360 expects."""
    df = X.copy()
    df["label"] = np.asarray(y)
    return BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=[sensitive_col],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )


def _groups(sensitive_col: str, privileged_value: int = 0) -> tuple[list[dict], list[dict]]:
    """Build the (privileged_groups, unprivileged_groups) dicts AIF360 wants."""
    unprivileged_value = 1 - privileged_value
    return (
        [{sensitive_col: privileged_value}],
        [{sensitive_col: unprivileged_value}],
    )


def _predicted_dataset(
    dataset_true: BinaryLabelDataset, y_pred: np.ndarray, y_scores: np.ndarray | None = None
) -> BinaryLabelDataset:
    """Clone a true-label dataset but swap in a model's predictions/scores."""
    dataset_pred = dataset_true.copy(deepcopy=True)
    dataset_pred.labels = np.asarray(y_pred).reshape(-1, 1)
    if y_scores is not None:
        dataset_pred.scores = np.asarray(y_scores).reshape(-1, 1)
    return dataset_pred


def _favorable_class_scores(model, X: pd.DataFrame, favorable_label: int = 0) -> np.ndarray:
    """
    Return P(favorable_label) per row, using whichever column index the
    model's `classes_` actually assigns to that label — not assumed to be
    column 1. AIF360's `BinaryLabelDataset.scores` must hold
    P(favorable outcome); passing P(the other class) by mistake silently
    inverts what these post-processing algorithms optimize for (this bit
    us in early testing: calibrated_equalized_odds was inverting every
    single prediction until this was fixed).
    """
    proba = model.predict_proba(X)
    class_idx = list(model.classes_).index(favorable_label)
    return proba[:, class_idx]


def reject_option_classification(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_col: str,
    privileged_value: int = 0,
    metric_name: str = "Statistical parity difference",
    low_class_thresh: float = 0.01,
    high_class_thresh: float = 0.99,
    num_class_thresh: int = 100,
    num_ROC_margin: int = 50,
) -> np.ndarray:
    """
    Reject Option Classification (Kamiran, Karim & Zhang, 2012).

    Shifts predictions near the decision boundary in favor of the
    unprivileged group, within a fairness-metric-defined margin.
    """
    from aif360.algorithms.postprocessing import RejectOptionClassification

    privileged_groups, unprivileged_groups = _groups(sensitive_col, privileged_value)
    dataset_true = _to_binary_label_dataset(X_test, y_test, sensitive_col)

    y_scores = _favorable_class_scores(model, X_test, favorable_label=0)
    y_pred = model.predict(X_test)
    dataset_pred = _predicted_dataset(dataset_true, y_pred, y_scores)

    roc = RejectOptionClassification(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        low_class_thresh=low_class_thresh,
        high_class_thresh=high_class_thresh,
        num_class_thresh=num_class_thresh,
        num_ROC_margin=num_ROC_margin,
        metric_name=metric_name,
    )
    roc.fit(dataset_true, dataset_pred)
    mitigated = roc.predict(dataset_pred)
    return mitigated.labels.ravel()


def equalized_odds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_col: str,
    privileged_value: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """
    Equalized Odds post-processing (Hardt, Price & Srebro, 2016).

    Randomly flips some predictions per group so that true/false positive
    rates are equalized across privileged/unprivileged groups.
    """
    from aif360.algorithms.postprocessing import EqOddsPostprocessing

    privileged_groups, unprivileged_groups = _groups(sensitive_col, privileged_value)
    dataset_true = _to_binary_label_dataset(X_test, y_test, sensitive_col)

    y_pred = model.predict(X_test)
    dataset_pred = _predicted_dataset(dataset_true, y_pred)

    eq = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        seed=seed,
    )
    eq.fit(dataset_true, dataset_pred)
    mitigated = eq.predict(dataset_pred)
    return mitigated.labels.ravel()


def calibrated_equalized_odds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_col: str,
    privileged_value: int = 0,
    cost_constraint: str = "fnr",
    seed: int = 42,
) -> np.ndarray:
    """
    Calibrated Equalized Odds post-processing (Pleiss et al., 2017).

    Like `equalized_odds`, but uses predicted *scores* (not just labels) so
    it can equalize a chosen error cost (`cost_constraint`: "fnr", "fpr",
    or "weighted") while keeping per-group score calibration.
    """
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

    privileged_groups, unprivileged_groups = _groups(sensitive_col, privileged_value)
    dataset_true = _to_binary_label_dataset(X_test, y_test, sensitive_col)

    y_scores = _favorable_class_scores(model, X_test, favorable_label=0)
    y_pred = model.predict(X_test)
    dataset_pred = _predicted_dataset(dataset_true, y_pred, y_scores)

    ceq = CalibratedEqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        cost_constraint=cost_constraint,
        seed=seed,
    )
    ceq.fit(dataset_true, dataset_pred)
    mitigated = ceq.predict(dataset_pred)
    return mitigated.labels.ravel()


def adversarial_debiasing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    sensitive_col: str,
    privileged_value: int = 0,
    num_epochs: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    scope_name: str = "fairkit_adversarial_debiasing",
) -> np.ndarray:
    """
    Adversarial Debiasing (Zhang, Lemoine & Mitchell, 2018) — IN-PROCESSING.

    Unlike the post-processing functions above, this trains its own neural
    classifier (a TensorFlow v1-graph model, internal to AIF360) jointly
    with an adversary that tries to predict the protected attribute from
    the classifier's output; the classifier is penalized when the
    adversary succeeds. It does NOT wrap `train_models()`'s output — call
    it directly on raw train/test splits instead.

    Note: uses `tensorflow.compat.v1` internally and needs eager execution
    disabled process-wide (handled here). Each call resets the default TF
    graph, so don't run this concurrently with other TF graph code.
    """
    import tensorflow.compat.v1 as tf1

    if tf1.executing_eagerly():
        tf1.disable_eager_execution()

    from aif360.algorithms.inprocessing import AdversarialDebiasing

    privileged_groups, unprivileged_groups = _groups(sensitive_col, privileged_value)
    dataset_train = _to_binary_label_dataset(X_train, y_train, sensitive_col)
    dataset_test = _to_binary_label_dataset(X_test, pd.Series(np.zeros(len(X_test)), index=X_test.index), sensitive_col)

    tf1.reset_default_graph()
    sess = tf1.Session()
    try:
        debiased_model = AdversarialDebiasing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            scope_name=scope_name,
            sess=sess,
            seed=seed,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        debiased_model.fit(dataset_train)
        mitigated = debiased_model.predict(dataset_test)
        return mitigated.labels.ravel()
    finally:
        sess.close()


# Registry so callers (e.g. the CLI) can dispatch mitigation by name.
POSTPROCESSING_TECHNIQUES = {
    "reject_option_classification": reject_option_classification,
    "equalized_odds": equalized_odds,
    "calibrated_equalized_odds": calibrated_equalized_odds,
}

INPROCESSING_TECHNIQUES = {
    "adversarial_debiasing": adversarial_debiasing,
}

ALL_TECHNIQUES = {**POSTPROCESSING_TECHNIQUES, **INPROCESSING_TECHNIQUES}
