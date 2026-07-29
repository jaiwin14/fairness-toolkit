import pandas as pd
import pytest

from fairkit.datasets.compas import (
    SENSITIVE_ATTRS,
    TARGET_COLUMNS,
    load_compas,
)


@pytest.fixture(scope="module")
def loaded():
    return load_compas()


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
    # sex_Male should always be present in this dataset
    assert "sex_Male" in sensitive_attrs


def test_targets_are_binary(loaded):
    _, y_dict, _ = loaded
    for target in TARGET_COLUMNS:
        values = set(y_dict[target].dropna().unique().tolist())
        assert values.issubset({0, 1}), f"{target} is not binary: {values}"


def test_raw_identifier_and_leakage_columns_dropped(loaded):
    X, _, _ = loaded
    dropped_examples = ["id", "name", "dob", "c_jail_in", "c_jail_out", "decile_score"]
    for col in dropped_examples:
        assert col not in X.columns
