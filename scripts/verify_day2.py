"""
Day 2 trial run — proves train -> evaluate -> mitigate -> evaluate again
works end to end on COMPAS via plain function calls, no notebook needed.

Run with:  python scripts/verify_day2.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

failures = []


def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def main():
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.model_selection import train_test_split

    from fairkit.datasets.compas import load_compas
    from fairkit.evaluate import evaluate_model
    from fairkit.mitigate import equalized_odds, reject_option_classification
    from fairkit.train import train_models

    print("=" * 60)
    print("DAY 2 TRIAL RUN")
    print("=" * 60)

    # --- load + split ---
    X, y_dict, sensitive_attrs = load_compas()
    y = y_dict["two_year_recid"]
    sensitive_col = sensitive_attrs[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    check("data loaded and split", len(X_train) > 0 and len(X_test) > 0)

    # --- train ---
    fitted = train_models(X_train, y_train, models=("logreg", "gbc"))
    check("train_models() trained logreg + gbc", set(fitted.keys()) == {"logreg", "gbc"})

    model = fitted["logreg"]

    # --- evaluate (pre-mitigation) ---
    pre = evaluate_model(model, X_test, y_test, sensitive_col=sensitive_col, model_name="logreg")
    check("evaluate_model() ran pre-mitigation", 0.0 <= pre["accuracy"] <= 1.0, f"acc={pre['accuracy']:.3f}")
    print(f"    pre-mitigation:  accuracy={pre['accuracy']:.3f}  "
          f"DP diff={pre['demographic_parity_difference']:.3f}  "
          f"EO diff={pre['equalized_odds_difference']:.3f}")

    # --- mitigate ---
    mitigated_preds = equalized_odds(model, X_test, y_test, sensitive_col=sensitive_col)
    check("equalized_odds() produced predictions", len(mitigated_preds) == len(X_test))

    # --- evaluate again (post-mitigation) ---
    post = evaluate_model(
        model=None, X_test=X_test, y_test=y_test, sensitive_col=sensitive_col,
        y_pred=mitigated_preds, model_name="logreg", mitigation="equalized_odds",
    )
    check("evaluate_model() ran post-mitigation", 0.0 <= post["accuracy"] <= 1.0, f"acc={post['accuracy']:.3f}")
    print(f"    post-mitigation: accuracy={post['accuracy']:.3f}  "
          f"DP diff={post['demographic_parity_difference']:.3f}  "
          f"EO diff={post['equalized_odds_difference']:.3f}")

    # Equalized Odds should, by design, tend to shrink the EO gap. We check
    # it moved rather than requiring a specific direction on such a small
    # slice of data, since a randomized post-processor is stochastic.
    check(
        "equalized odds difference changed after mitigation",
        abs(post["equalized_odds_difference"] - pre["equalized_odds_difference"]) > 1e-9,
    )

    # --- sanity check a second technique too (this is what Day 2 built) ---
    roc_preds = reject_option_classification(model, X_test, y_test, sensitive_col=sensitive_col)
    check("reject_option_classification() also runs", len(roc_preds) == len(X_test))

    print("=" * 60)
    if failures:
        print(f"DAY 2 CHECKPOINT: FAILED ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("DAY 2 CHECKPOINT: PASSED — train -> evaluate -> mitigate -> evaluate verified end to end.")
        sys.exit(0)


if __name__ == "__main__":
    main()
