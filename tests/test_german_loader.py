import pandas as pd
import pytest

from fairkit.datasets.german import (
    FAVORABLE_LABEL,
    SENSITIVE_ATTRS,
    TARGET_COLUMNS,
    load_german,
)


@pytest.fixture(scope="module")
def loaded():
    return load_german()


def test_returns_expected_types(loaded):
    X, y_dict, sensitive_attrs = loaded
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y_dict, dict)
    assert isinstance(sensitive_attrs, list)


def test_row_count_matches_known_dataset_size(loaded):
    # UCI German Credit is exactly 1000 instances, no missing values.
    X, _, _ = loaded
    assert X.shape[0] == 1000


def test_no_missing_values_in_features(loaded):
    X, _, _ = loaded
    assert not X.isnull().any().any()


def test_features_are_fully_numeric(loaded):
    X, _, _ = loaded
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns leaked through: {non_numeric}"


def test_sensitive_columns_exist(loaded):
    X, _, sensitive_attrs = loaded
    assert set(sensitive_attrs) == set(SENSITIVE_ATTRS)
    for col in sensitive_attrs:
        assert col in X.columns


def test_target_is_binary(loaded):
    _, y_dict, _ = loaded
    values = set(y_dict["credit_risk"].dropna().unique().tolist())
    assert values.issubset({0, 1})


def test_good_credit_rate_matches_known_class_split(loaded):
    # UCI documents the class split as exactly 700 good / 300 bad.
    _, y_dict, _ = loaded
    rate = y_dict["credit_risk"].mean()
    assert abs(rate - 0.70) < 0.01, f"got {rate:.3f}"


def test_favorable_label_matches_adult_not_compas(loaded):
    assert FAVORABLE_LABEL == 1


def test_personal_status_and_class_columns_absent(loaded):
    X, _, _ = loaded
    assert "personal_status" not in X.columns
    assert "class" not in X.columns
