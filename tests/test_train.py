import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fairkit.datasets.compas import load_compas
from fairkit.evaluate import evaluate_model
from fairkit.train import AVAILABLE_MODELS, train_models


@pytest.fixture(scope="module")
def compas_split():
    X, y_dict, sensitive_attrs = load_compas()
    y = y_dict["two_year_recid"]
    # small, fast split — this is a smoke test, not a benchmark run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, sensitive_attrs


def test_train_models_returns_all_requested_models(compas_split):
    X_train, _, y_train, _, _ = compas_split
    fitted = train_models(X_train, y_train, models=("logreg", "gbc"))
    assert set(fitted.keys()) == {"logreg", "gbc"}


def test_train_models_default_trains_everything(compas_split):
    X_train, _, y_train, _, _ = compas_split
    # subsample for speed — smoke test only needs to prove it runs
    X_small, y_small = X_train.iloc[:300], y_train.iloc[:300]
    fitted = train_models(X_small, y_small)
    assert set(fitted.keys()) == set(AVAILABLE_MODELS)
    for name, model in fitted.items():
        assert hasattr(model, "predict"), f"{name} missing .predict"


def test_train_models_rejects_unknown_model_name(compas_split):
    X_train, _, y_train, _, _ = compas_split
    with pytest.raises(ValueError):
        train_models(X_train.iloc[:50], y_train.iloc[:50], models=("not_a_real_model",))


def test_evaluate_model_returns_expected_keys(compas_split):
    X_train, X_test, y_train, y_test, sensitive_attrs = compas_split
    fitted = train_models(X_train.iloc[:300], y_train.iloc[:300], models=("logreg",))
    result = evaluate_model(
        fitted["logreg"], X_test, y_test, sensitive_col=sensitive_attrs[0], model_name="logreg"
    )
    expected_keys = {
        "model_name", "dataset_name", "mitigation", "sensitive_col",
        "accuracy", "demographic_parity_difference",
        "equalized_odds_difference", "accuracy_by_group",
    }
    assert expected_keys.issubset(result.keys())
    assert 0.0 <= result["accuracy"] <= 1.0
    assert isinstance(result["accuracy_by_group"], dict)


def test_evaluate_model_accepts_precomputed_predictions(compas_split):
    X_train, X_test, y_train, y_test, sensitive_attrs = compas_split
    fitted = train_models(X_train.iloc[:300], y_train.iloc[:300], models=("logreg",))
    y_pred = fitted["logreg"].predict(X_test)
    result = evaluate_model(
        model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_attrs[0], y_pred=y_pred
    )
    assert 0.0 <= result["accuracy"] <= 1.0
