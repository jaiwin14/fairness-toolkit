"""
Day 5 trial run — verifies the German Credit loader, the German benchmark
results, and the cross-dataset comparison (the actual novel contribution
of this day) all exist and are internally consistent.

Run with:  python scripts/verify_day5.py
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

    import pandas as pd

    print("=" * 60)
    print("DAY 5 TRIAL RUN")
    print("=" * 60)

    from fairkit.datasets.german import FAVORABLE_LABEL, load_german

    # --- loader sanity ---
    X, y_dict, sensitive_attrs = load_german()
    check("load_german() runs without error", True)
    check("German row count matches known dataset size (1000)", X.shape[0] == 1000, f"got {X.shape[0]}")
    check("no NaNs in German features", not X.isnull().any().any())
    check("German sensitive attributes present", set(sensitive_attrs) == {"sex_Male", "age_ge_25"})

    good_rate = y_dict["credit_risk"].mean()
    check(
        "good-credit rate matches known 700/300 class split",
        abs(good_rate - 0.70) < 0.01,
        f"got {good_rate:.3f}",
    )

    # --- German results file ---
    german_csv = REPO_ROOT / "results" / "german_results.csv"
    check("results/german_results.csv exists", german_csv.exists())

    if german_csv.exists():
        gdf = pd.read_csv(german_csv)
        check("German results: 17 total rows", len(gdf) == 17, f"got {len(gdf)}")
        baseline = gdf[gdf["mitigation"] == "none"]
        check(
            "German baseline accuracies in a sane range (0.60-0.85)",
            baseline["accuracy"].between(0.60, 0.85).all(),
        )

    # --- cross-dataset comparison ---
    cross_csv = REPO_ROOT / "results" / "cross_dataset_comparison.csv"
    cross_png = REPO_ROOT / "results" / "cross_dataset_comparison.png"
    check("results/cross_dataset_comparison.csv exists", cross_csv.exists())
    check("results/cross_dataset_comparison.png exists", cross_png.exists())

    if cross_csv.exists():
        cdf = pd.read_csv(cross_csv)
        check(
            "cross-dataset comparison covers all 3 datasets",
            set(cdf["dataset"].unique()) == {"compas", "adult", "german"},
        )
        check(
            "cross-dataset comparison covers all 4 models",
            {"logreg", "svc", "gbc", "xgboost"}.issubset(set(cdf["model"].unique())),
        )
        check("36 total rows (3 datasets x 4 models x 3 mitigations)", len(cdf) == 36, f"got {len(cdf)}")

        # The actual headline finding: equalized_odds should never make
        # fairness worse, across every dataset/model combination tested.
        eo = cdf[cdf["mitigation"] == "equalized_odds"]
        check(
            "equalized_odds never worsens fairness (di_gap_improvement >= 0 everywhere)",
            (eo["di_gap_improvement"] >= -1e-9).all(),
            f"worst case was {eo['di_gap_improvement'].min():.3f}",
        )

    readme = (REPO_ROOT / "README.md").read_text()
    check("README contains a German Credit results section", "## Results: German Credit" in readme)
    check("README contains a cross-dataset comparison section", "## Cross-dataset comparison" in readme)

    print("=" * 60)
    if failures:
        print(f"DAY 5 CHECKPOINT: FAILED ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("DAY 5 CHECKPOINT: PASSED — German Credit + cross-dataset comparison verified end to end.")
        sys.exit(0)


if __name__ == "__main__":
    main()
