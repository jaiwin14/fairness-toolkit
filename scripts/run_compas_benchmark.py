"""
Full COMPAS benchmark: all models x both recidivism targets x
baseline + each mitigation technique.

Run with:  python scripts/run_compas_benchmark.py

Writes results/compas_results.csv — one row per (target, model, mitigation)
combination, with accuracy + fairness metrics from fairkit.evaluate.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from fairkit.datasets.compas import load_compas
from fairkit.evaluate import evaluate_model
from fairkit.mitigate import (
    adversarial_debiasing,
    calibrated_equalized_odds,
    equalized_odds,
    reject_option_classification,
)
from fairkit.train import AVAILABLE_MODELS, train_models

TARGETS = ["is_recid", "two_year_recid"]
RANDOM_STATE = 42

POSTPROCESSING = {
    "reject_option_classification": reject_option_classification,
    "equalized_odds": equalized_odds,
    "calibrated_equalized_odds": calibrated_equalized_odds,
}


def run_for_target(target: str) -> list[dict]:
    print(f"\n=== target: {target} ===")
    X, y_dict, sensitive_attrs = load_compas()
    y = y_dict[target]
    sensitive_col = sensitive_attrs[0]  # sex_Male

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    fitted = train_models(X_train, y_train, models=AVAILABLE_MODELS, random_state=RANDOM_STATE)

    rows = []

    # baseline (no mitigation) + each post-processing technique, per model
    for model_name, model in fitted.items():
        baseline = evaluate_model(
            model, X_test, y_test, sensitive_col=sensitive_col,
            model_name=model_name, dataset_name="compas", mitigation="none",
        )
        baseline["target"] = target
        rows.append(baseline)
        print(f"  [{model_name:8s}] baseline           "
              f"acc={baseline['accuracy']:.3f}  "
              f"DPdiff={baseline['demographic_parity_difference']:.3f}  "
              f"EOdiff={baseline['equalized_odds_difference']:.3f}")

        for mit_name, mit_fn in POSTPROCESSING.items():
            try:
                y_pred_mit = mit_fn(model, X_test, y_test, sensitive_col=sensitive_col)
                result = evaluate_model(
                    model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
                    y_pred=y_pred_mit, model_name=model_name, dataset_name="compas",
                    mitigation=mit_name,
                )
            except Exception as e:
                result = {
                    "model_name": model_name, "dataset_name": "compas", "mitigation": mit_name,
                    "sensitive_col": sensitive_col, "accuracy": None,
                    "demographic_parity_difference": None, "equalized_odds_difference": None,
                    "disparate_impact_ratio": None, "accuracy_by_group": None, "error": str(e),
                }
            result["target"] = target
            rows.append(result)
            acc_str = f"{result['accuracy']:.3f}" if result["accuracy"] is not None else "FAILED"
            print(f"  [{model_name:8s}] {mit_name:28s} acc={acc_str}")

    # adversarial debiasing: model-agnostic (trains its own classifier), one row per target
    try:
        y_pred_adv = adversarial_debiasing(
            X_train, y_train, X_test, sensitive_col=sensitive_col, num_epochs=20, seed=RANDOM_STATE
        )
        adv_result = evaluate_model(
            model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
            y_pred=y_pred_adv, model_name="adversarial_debiasing_nn", dataset_name="compas",
            mitigation="adversarial_debiasing",
        )
    except Exception as e:
        adv_result = {
            "model_name": "adversarial_debiasing_nn", "dataset_name": "compas",
            "mitigation": "adversarial_debiasing", "sensitive_col": sensitive_col,
            "accuracy": None, "demographic_parity_difference": None,
            "equalized_odds_difference": None, "disparate_impact_ratio": None,
            "accuracy_by_group": None, "error": str(e),
        }
    adv_result["target"] = target
    rows.append(adv_result)
    acc_str = f"{adv_result['accuracy']:.3f}" if adv_result["accuracy"] is not None else "FAILED"
    print(f"  [adv_debias] adversarial_debiasing         acc={acc_str}")

    return rows


def main():
    all_rows = []
    for target in TARGETS:
        all_rows.extend(run_for_target(target))

    df = pd.DataFrame(all_rows)
    out_path = REPO_ROOT / "results" / "compas_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
