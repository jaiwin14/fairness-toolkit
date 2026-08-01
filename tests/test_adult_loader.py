import pandas as pd
import pytest

from fairkit.datasets.adult import (
    FAVORABLE_LABEL,
    SENSITIVE_ATTRS,
    TARGET_COLUMNS,
    load_adult,
)


@pytest.fixture(scope="module")
def loaded():
    return load_adult()


def test_returns_expected_types(loaded):
    X, y_dict, sensitive_attrs = loaded
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y_dict, dict)
    assert isinstance(sensitive_attrs, list)


def test_shape_is_nonempty(loaded):
    X, y_dict, _ = loaded
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    for target in TARGET_COLUMNS:
        assert y_dict[target].shape[0] == X.shape[0]


def test_row_count_matches_known_cleaned_adult(loaded):
    # Well-documented figure for Adult after dropping rows with missing
    # values: 30,162 (from the original 32,561). Good regression check.
    X, _, _ = loaded
    assert X.shape[0] == 30162


def test_no_missing_values_in_features(loaded):
    X, _, _ = loaded
    assert not X.isnull().any().any(), "Feature matrix contains NaNs"


def test_features_are_fully_numeric(loaded):
    X, _, _ = loaded
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns leaked through: {non_numeric}"


def test_sensitive_columns_exist(loaded):
    X, _, sensitive_attrs = loaded
    assert len(sensitive_attrs) > 0
    for col in sensitive_attrs:
        assert col in X.columns
    assert set(sensitive_attrs) == set(SENSITIVE_ATTRS)


def test_target_is_binary(loaded):
    _, y_dict, _ = loaded
    values = set(y_dict["income"].dropna().unique().tolist())
    assert values.issubset({0, 1})


def test_income_rate_matches_known_baseline(loaded):
    # Published baseline positive rate (>$50K) for cleaned Adult is ~24%.
    _, y_dict, _ = loaded
    rate = y_dict["income"].mean()
    assert 0.20 < rate < 0.30, f"got {rate:.3f}"


def test_favorable_label_is_one_unlike_compas(loaded):
    # This is the whole point of Day 4: Adult's favorable outcome (income
    # >50K) is label 1, the OPPOSITE convention from COMPAS (favorable=0).
    # Downstream code must read this from the dataset module, not assume.
    assert FAVORABLE_LABEL == 1


def test_dropped_columns_absent(loaded):
    X, _, _ = loaded
    for col in ["fnlwgt", "education", "native_country"]:
        assert col not in X.columns
