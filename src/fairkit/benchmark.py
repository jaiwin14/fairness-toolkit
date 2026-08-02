"""
Generic, dataset-agnostic benchmark runner.

This is the CLI's engine (`fairkit benchmark`). It's a single generalized
implementation of the same logic each day-specific script
(`scripts/run_compas_benchmark.py`, `run_adult_benchmark.py`,
`run_german_benchmark.py`) hand-wrote for its own dataset — those scripts
stay as-is (they're each day's actual historical work and read clearly on
their own), but new dataset-agnostic usage should go through this instead
of copy-pasting a fourth near-identical script.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluate import evaluate_model
from .mitigate import ALL_TECHNIQUES, POSTPROCESSING_TECHNIQUES, adversarial_debiasing
from .registry import get_dataset_spec
from .train import AVAILABLE_MODELS, train_models

RANDOM_STATE = 42


def run_benchmark(
    dataset_name: str,
    models: tuple[str, ...] = AVAILABLE_MODELS,
    include_adversarial_debiasing: bool = True,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train each requested model on `dataset_name`, evaluate baseline +
    every post-processing mitigation technique, and (optionally)
    adversarial debiasing. Returns one row per (model, mitigation).
    """
    warnings.filterwarnings("ignore")
    spec = get_dataset_spec(dataset_name)

    X, y_dict, sensitive_attrs = spec.loader()
    y = y_dict[spec.default_target]
    sensitive_col = sensitive_attrs[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    fitted = train_models(X_train, y_train, models=models, random_state=random_state)

    common_kwargs = dict(
        sensitive_col=sensitive_col,
        privileged_value=spec.privileged_value,
        favorable_label=spec.favorable_label,
        unfavorable_label=spec.unfavorable_label,
    )

    rows = []
    for model_name, model in fitted.items():
        baseline = evaluate_model(
            model, X_test, y_test, sensitive_col=sensitive_col,
            model_name=model_name, dataset_name=dataset_name, mitigation="none",
            favorable_label=spec.favorable_label,
        )
        rows.append(baseline)
        if verbose:
            print(f"  [{model_name:8s}] baseline           acc={baseline['accuracy']:.3f}")

        for mit_name, mit_fn in POSTPROCESSING_TECHNIQUES.items():
            try:
                y_pred = mit_fn(model, X_test, y_test, **common_kwargs)
                result = evaluate_model(
                    model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
                    y_pred=y_pred, model_name=model_name, dataset_name=dataset_name,
                    mitigation=mit_name, favorable_label=spec.favorable_label,
                )
            except Exception as e:
                result = {
                    "model_name": model_name, "dataset_name": dataset_name, "mitigation": mit_name,
                    "sensitive_col": sensitive_col, "accuracy": None,
                    "demographic_parity_difference": None, "equalized_odds_difference": None,
                    "disparate_impact_ratio": None, "accuracy_by_group": None, "error": str(e),
                }
            rows.append(result)
            if verbose:
                acc_str = f"{result['accuracy']:.3f}" if result["accuracy"] is not None else "FAILED"
                print(f"  [{model_name:8s}] {mit_name:28s} acc={acc_str}")

    if include_adversarial_debiasing:
        try:
            y_pred_adv = adversarial_debiasing(X_train, y_train, X_test, num_epochs=20, **common_kwargs)
            adv_result = evaluate_model(
                model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
                y_pred=y_pred_adv, model_name="adversarial_debiasing_nn", dataset_name=dataset_name,
                mitigation="adversarial_debiasing", favorable_label=spec.favorable_label,
            )
        except Exception as e:
            adv_result = {
                "model_name": "adversarial_debiasing_nn", "dataset_name": dataset_name,
                "mitigation": "adversarial_debiasing", "sensitive_col": sensitive_col,
                "accuracy": None, "demographic_parity_difference": None,
                "equalized_odds_difference": None, "disparate_impact_ratio": None,
                "accuracy_by_group": None, "error": str(e),
            }
        rows.append(adv_result)
        if verbose:
            acc_str = f"{adv_result['accuracy']:.3f}" if adv_result["accuracy"] is not None else "FAILED"
            print(f"  [adv_debias] adversarial_debiasing         acc={acc_str}")

    return pd.DataFrame(rows)


def run_single(
    dataset_name: str,
    model_name: str,
    mitigation: str = "none",
    random_state: int = RANDOM_STATE,
) -> dict:
    """Train one model on one dataset, apply one mitigation, return one result row."""
    spec = get_dataset_spec(dataset_name)

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(AVAILABLE_MODELS)}")
    if mitigation != "none" and mitigation not in ALL_TECHNIQUES:
        raise ValueError(f"Unknown mitigation '{mitigation}'. Available: none, {list(ALL_TECHNIQUES)}")

    X, y_dict, sensitive_attrs = spec.loader()
    y = y_dict[spec.default_target]
    sensitive_col = sensitive_attrs[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    fitted = train_models(X_train, y_train, models=(model_name,), random_state=random_state)
    model = fitted[model_name]

    common_kwargs = dict(
        sensitive_col=sensitive_col,
        privileged_value=spec.privileged_value,
        favorable_label=spec.favorable_label,
        unfavorable_label=spec.unfavorable_label,
    )

    if mitigation == "none":
        return evaluate_model(
            model, X_test, y_test, sensitive_col=sensitive_col,
            model_name=model_name, dataset_name=dataset_name, mitigation="none",
            favorable_label=spec.favorable_label,
        )
    elif mitigation == "adversarial_debiasing":
        y_pred = adversarial_debiasing(X_train, y_train, X_test, num_epochs=20, **common_kwargs)
        model_label = "adversarial_debiasing_nn"
    else:
        mit_fn = POSTPROCESSING_TECHNIQUES[mitigation]
        y_pred = mit_fn(model, X_test, y_test, **common_kwargs)
        model_label = model_name

    return evaluate_model(
        model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
        y_pred=y_pred, model_name=model_label, dataset_name=dataset_name,
        mitigation=mitigation, favorable_label=spec.favorable_label,
    )
