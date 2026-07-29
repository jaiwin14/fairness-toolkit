import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fairkit.datasets.compas import load_compas
from fairkit.mitigate import (
    adversarial_debiasing,
    calibrated_equalized_odds,
    equalized_odds,
    reject_option_classification,
)


@pytest.fixture(scope="module")
def small_split():
    """A small subsample — these are smoke tests for correctness, not scale."""
    X, y_dict, sensitive_attrs = load_compas()
    y = y_dict["two_year_recid"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    sensitive_col = sensitive_attrs[0]
    return (
        X_train.iloc[:400], X_test.iloc[:200],
        y_train.iloc[:400], y_test.iloc[:200],
        sensitive_col,
    )


@pytest.fixture(scope="module")
def fitted_logreg(small_split):
    X_train, _, y_train, _, _ = small_split
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def _assert_valid_binary_predictions(preds, expected_len):
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (expected_len,)
    assert set(np.unique(preds)).issubset({0.0, 1.0, 0, 1})


def test_reject_option_classification_runs(small_split, fitted_logreg):
    X_train, X_test, y_train, y_test, sensitive_col = small_split
    preds = reject_option_classification(fitted_logreg, X_test, y_test, sensitive_col)
    _assert_valid_binary_predictions(preds, len(X_test))


def test_equalized_odds_runs(small_split, fitted_logreg):
    X_train, X_test, y_train, y_test, sensitive_col = small_split
    preds = equalized_odds(fitted_logreg, X_test, y_test, sensitive_col)
    _assert_valid_binary_predictions(preds, len(X_test))


def test_calibrated_equalized_odds_runs(small_split, fitted_logreg):
    X_train, X_test, y_train, y_test, sensitive_col = small_split
    preds = calibrated_equalized_odds(fitted_logreg, X_test, y_test, sensitive_col)
    _assert_valid_binary_predictions(preds, len(X_test))


def test_adversarial_debiasing_runs(small_split):
    X_train, X_test, y_train, y_test, sensitive_col = small_split
    # tiny num_epochs — this only needs to prove the TF graph plumbing works
    preds = adversarial_debiasing(
        X_train, y_train, X_test, sensitive_col, num_epochs=3, batch_size=64
    )
    _assert_valid_binary_predictions(preds, len(X_test))
