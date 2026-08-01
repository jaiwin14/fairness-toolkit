"""
Full German Credit benchmark: all models x baseline + each mitigation
technique. Third and final dataset through the identical pipeline used
for COMPAS and Adult.

Run with:  python scripts/run_german_benchmark.py

Writes results/german_results.csv.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from fairkit.datasets.german import FAVORABLE_LABEL, UNFAVORABLE_LABEL, load_german
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

# sex_Male=1 means "is Male" directly (same encoding style as Adult's
# race_White), so privileged_value=1 here too.
PRIVILEGED_VALUE = 1


def main():
    print("=== dataset: german ===")
    X, y_dict, sensitive_attrs = load_german()
    y = y_dict["credit_risk"]
    sensitive_col = sensitive_attrs[0]  # sex_Male

    # German Credit is only 1000 rows -- a 0.2 test split leaves 200 test
    # rows, already small; stratify to keep the 70/30 class balance intact.
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
            model_name=model_name, dataset_name="german", mitigation="none",
            favorable_label=FAVORABLE_LABEL,
        )
        baseline["target"] = "credit_risk"
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
                    y_pred=y_pred_mit, model_name=model_name, dataset_name="german",
                    mitigation=mit_name, favorable_label=FAVORABLE_LABEL,
                )
            except Exception as e:
                result = {
                    "model_name": model_name, "dataset_name": "german", "mitigation": mit_name,
                    "sensitive_col": sensitive_col, "accuracy": None,
                    "demographic_parity_difference": None, "equalized_odds_difference": None,
                    "disparate_impact_ratio": None, "accuracy_by_group": None, "error": str(e),
                }
            result["target"] = "credit_risk"
            rows.append(result)
            acc_str = f"{result['accuracy']:.3f}" if result["accuracy"] is not None else "FAILED"
            print(f"  [{model_name:8s}] {mit_name:28s} acc={acc_str}")

    try:
        y_pred_adv = adversarial_debiasing(
            X_train, y_train, X_test, num_epochs=20, **common_kwargs
        )
        adv_result = evaluate_model(
            model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
            y_pred=y_pred_adv, model_name="adversarial_debiasing_nn", dataset_name="german",
            mitigation="adversarial_debiasing", favorable_label=FAVORABLE_LABEL,
        )
    except Exception as e:
        adv_result = {
            "model_name": "adversarial_debiasing_nn", "dataset_name": "german",
            "mitigation": "adversarial_debiasing", "sensitive_col": sensitive_col,
            "accuracy": None, "demographic_parity_difference": None,
            "equalized_odds_difference": None, "disparate_impact_ratio": None,
            "accuracy_by_group": None, "error": str(e),
        }
    adv_result["target"] = "credit_risk"
    rows.append(adv_result)
    acc_str = f"{adv_result['accuracy']:.3f}" if adv_result["accuracy"] is not None else "FAILED"
    print(f"  [adv_debias] adversarial_debiasing         acc={acc_str}")

    df = pd.DataFrame(rows)
    out_path = REPO_ROOT / "results" / "german_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
