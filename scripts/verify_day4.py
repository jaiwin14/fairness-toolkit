"""
Day 4 trial run — verifies the Adult Income loader works, and that the
SAME train/evaluate/mitigate pipeline used for COMPAS produces sane,
non-degenerate results on a dataset with the opposite favorable-label
convention. That's the actual generalization test for this day's work.

Run with:  python scripts/verify_day4.py
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
    print("DAY 4 TRIAL RUN")
    print("=" * 60)

    from fairkit.datasets.adult import FAVORABLE_LABEL, load_adult

    # --- loader sanity ---
    X, y_dict, sensitive_attrs = load_adult()
    check("load_adult() runs without error", True)
    check("Adult row count matches known cleaned figure (30162)", X.shape[0] == 30162, f"got {X.shape[0]}")
    check("no NaNs in Adult features", not X.isnull().any().any())
    check("Adult sensitive attributes present", set(sensitive_attrs) == {"sex_Male", "race_White"})
    check(
        "Adult's favorable label is opposite of COMPAS's (generalization check)",
        FAVORABLE_LABEL == 1,
    )

    # --- results file sanity ---
    csv_path = REPO_ROOT / "results" / "adult_results.csv"
    check("results/adult_results.csv exists", csv_path.exists())

    if not csv_path.exists():
        print("=" * 60)
        print("DAY 4 CHECKPOINT: FAILED — no results CSV to check further.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    check(
        "results cover all 4 base models",
        {"logreg", "svc", "gbc", "xgboost"}.issubset(set(df["model_name"].unique())),
    )
    check(
        "results cover all 4 mitigation states",
        {"none", "reject_option_classification", "equalized_odds",
         "calibrated_equalized_odds", "adversarial_debiasing"}.issubset(set(df["mitigation"].unique())),
    )
    check("17 total rows (4 models x 4 states + 1 adversarial)", len(df) == 17, f"got {len(df)}")

    baseline = df[df["mitigation"] == "none"]
    check(
        "baseline accuracies in a sane published-benchmark range (0.75-0.90)",
        baseline["accuracy"].between(0.75, 0.90).all(),
        f"range was {baseline['accuracy'].min():.3f}-{baseline['accuracy'].max():.3f}",
    )

    # The actual generalization proof: this exact bug class (scores passed
    # backwards for the wrong favorable_label) would show up as degenerate
    # accuracy here specifically, since Adult's convention is inverted
    # relative to COMPAS's. If this passes, the dataset-agnostic refactor
    # genuinely works, not just "happens to work for COMPAS's convention."
    mitigated = df[df["mitigation"] != "none"]
    check(
        "no mitigation technique collapses to degenerate accuracy (<0.65)",
        (mitigated["accuracy"] >= 0.65).all(),
        f"min mitigated accuracy was {mitigated['accuracy'].min():.3f}",
    )

    # Baseline disparate impact should show real, substantial bias here
    # (well-documented gender/race income gap) — a healthy sanity check
    # that favorable_label was threaded correctly end to end.
    check(
        "baseline disparate impact shows real bias (ratio far from 1.0)",
        (baseline["disparate_impact_ratio"] > 1.5).all(),
        f"min baseline DI ratio was {baseline['disparate_impact_ratio'].min():.3f}",
    )

    readme = (REPO_ROOT / "README.md").read_text()
    check("README contains an Adult Income results section", "<b>Adult Income</b>" in readme)

    print("=" * 60)
    if failures:
        print(f"DAY 4 CHECKPOINT: FAILED ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("DAY 4 CHECKPOINT: PASSED — Adult loader + generalized pipeline verified end to end.")
        sys.exit(0)


if __name__ == "__main__":
    main()
