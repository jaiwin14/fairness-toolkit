"""
Full Adult Income benchmark: all models x baseline + each mitigation
technique. Structurally identical to run_compas_benchmark.py — this script
is the proof that Days 1-3's pipeline is genuinely dataset-agnostic, not
just renamed COMPAS code.

Run with:  python scripts/run_adult_benchmark.py

Writes results/adult_results.csv.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from fairkit.datasets.adult import FAVORABLE_LABEL, UNFAVORABLE_LABEL, load_adult
from fairkit.evaluate import evaluate_model
from fairkit.mitigate import (
    adversarial_debiasing,
    calibrated_equalized_odds,
    equalized_odds,
    reject_option_classification,
)
from fairkit.train import AVAILABLE_MODELS, train_models

RANDOM_STATE = 42

POSTPROCESSING = {
    "reject_option_classification": reject_option_classification,
    "equalized_odds": equalized_odds,
    "calibrated_equalized_odds": calibrated_equalized_odds,
}

# Adult's column encodes the PRIVILEGED group directly (sex_Male=1 means
# "is Male", race_White is analogous) — the opposite of COMPAS's
# race_African-American, which encodes the unprivileged group. So
# privileged_value=1 here, vs. the default 0 used for COMPAS.
PRIVILEGED_VALUE = 1


def main():
    print("=== dataset: adult ===")
    X, y_dict, sensitive_attrs = load_adult()
    y = y_dict["income"]
    sensitive_col = sensitive_attrs[0]  # sex_Male

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    fitted = train_models(X_train, y_train, models=AVAILABLE_MODELS, random_state=RANDOM_STATE)

    common_kwargs = dict(
        sensitive_col=sensitive_col,
        privileged_value=PRIVILEGED_VALUE,
        favorable_label=FAVORABLE_LABEL,
        unfavorable_label=UNFAVORABLE_LABEL,
    )

    rows = []

    for model_name, model in fitted.items():
        baseline = evaluate_model(
            model, X_test, y_test, sensitive_col=sensitive_col,
            model_name=model_name, dataset_name="adult", mitigation="none",
            favorable_label=FAVORABLE_LABEL,
        )
        baseline["target"] = "income"
        rows.append(baseline)
        print(f"  [{model_name:8s}] baseline           "
              f"acc={baseline['accuracy']:.3f}  "
              f"DPdiff={baseline['demographic_parity_difference']:.3f}  "
              f"EOdiff={baseline['equalized_odds_difference']:.3f}")

        for mit_name, mit_fn in POSTPROCESSING.items():
            try:
                y_pred_mit = mit_fn(model, X_test, y_test, **common_kwargs)
                result = evaluate_model(
                    model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
                    y_pred=y_pred_mit, model_name=model_name, dataset_name="adult",
                    mitigation=mit_name, favorable_label=FAVORABLE_LABEL,
                )
            except Exception as e:
                result = {
                    "model_name": model_name, "dataset_name": "adult", "mitigation": mit_name,
                    "sensitive_col": sensitive_col, "accuracy": None,
                    "demographic_parity_difference": None, "equalized_odds_difference": None,
                    "disparate_impact_ratio": None, "accuracy_by_group": None, "error": str(e),
                }
            result["target"] = "income"
            rows.append(result)
            acc_str = f"{result['accuracy']:.3f}" if result["accuracy"] is not None else "FAILED"
            print(f"  [{model_name:8s}] {mit_name:28s} acc={acc_str}")

    # adversarial debiasing: model-agnostic, one row
    try:
        y_pred_adv = adversarial_debiasing(
            X_train, y_train, X_test, num_epochs=20, **common_kwargs
        )
        adv_result = evaluate_model(
            model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
            y_pred=y_pred_adv, model_name="adversarial_debiasing_nn", dataset_name="adult",
            mitigation="adversarial_debiasing", favorable_label=FAVORABLE_LABEL,
        )
    except Exception as e:
        adv_result = {
            "model_name": "adversarial_debiasing_nn", "dataset_name": "adult",
            "mitigation": "adversarial_debiasing", "sensitive_col": sensitive_col,
            "accuracy": None, "demographic_parity_difference": None,
            "equalized_odds_difference": None, "disparate_impact_ratio": None,
            "accuracy_by_group": None, "error": str(e),
        }
    adv_result["target"] = "income"
    rows.append(adv_result)
    acc_str = f"{adv_result['accuracy']:.3f}" if adv_result["accuracy"] is not None else "FAILED"
    print(f"  [adv_debias] adversarial_debiasing         acc={acc_str}")

    df = pd.DataFrame(rows)
    out_path = REPO_ROOT / "results" / "adult_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
