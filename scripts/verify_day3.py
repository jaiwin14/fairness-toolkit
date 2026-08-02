"""
Day 3 trial run — verifies the full COMPAS benchmark + tradeoff chart +
README results section are all present, consistent, and non-degenerate.

Run with:  python scripts/verify_day3.py
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
    import pandas as pd

    print("=" * 60)
    print("DAY 3 TRIAL RUN")
    print("=" * 60)

    csv_path = REPO_ROOT / "results" / "compas_results.csv"
    png_path = REPO_ROOT / "results" / "compas_tradeoff.png"

    check("results/compas_results.csv exists", csv_path.exists())
    check("results/compas_tradeoff.png exists", png_path.exists())

    if not csv_path.exists():
        print("=" * 60)
        print("DAY 3 CHECKPOINT: FAILED — no results CSV to check further.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    expected_cols = {
        "model_name", "mitigation", "accuracy", "demographic_parity_difference",
        "equalized_odds_difference", "disparate_impact_ratio", "target",
    }
    check("results CSV has expected columns", expected_cols.issubset(df.columns))

    check(
        "results cover both targets",
        set(df["target"].unique()) == {"is_recid", "two_year_recid"},
        f"got {df['target'].unique().tolist()}",
    )
    check(
        "results cover all 4 base models",
        {"logreg", "svc", "gbc", "xgboost"}.issubset(set(df["model_name"].unique())),
    )
    check(
        "results cover all 4 mitigation techniques (+ baseline)",
        {"none", "reject_option_classification", "equalized_odds",
         "calibrated_equalized_odds", "adversarial_debiasing"}.issubset(set(df["mitigation"].unique())),
    )
    check("34 total rows (2 targets x (4 models x 4 states + 1 adversarial))", len(df) == 34, f"got {len(df)}")

    # Regression check: baseline accuracy should be in the same ballpark as
    # the original notebooks (which reported ~0.66-0.70 across models).
    baseline = df[df["mitigation"] == "none"]
    check(
        "baseline accuracies match original notebook ballpark (0.60-0.72)",
        baseline["accuracy"].between(0.60, 0.72).all(),
        f"range was {baseline['accuracy'].min():.3f}-{baseline['accuracy'].max():.3f}",
    )

    # Non-degeneracy check: no mitigation technique should produce accuracy
    # far below what a trivial majority-class classifier would get (this is
    # exactly the bug this project hit and fixed on Day 3 — a scores-
    # direction mistake that made calibrated_equalized_odds invert every
    # prediction). Majority class rate is always >= 0.5 for a binary target,
    # so healthy mitigated accuracy shouldn't be too far under that.
    mitigated = df[df["mitigation"] != "none"]
    check(
        "no mitigation technique collapses to degenerate accuracy (<0.55)",
        (mitigated["accuracy"] >= 0.55).all(),
        f"min mitigated accuracy was {mitigated['accuracy'].min():.3f}",
    )

    # README should contain a real, non-placeholder results section.
    readme = (REPO_ROOT / "README.md").read_text()
    check("README contains a COMPAS results section", "<b>COMPAS</b>" in readme)
    check("README results table is populated (not just a placeholder)", "| is_recid |" in readme)

    print("=" * 60)
    if failures:
        print(f"DAY 3 CHECKPOINT: FAILED ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("DAY 3 CHECKPOINT: PASSED — benchmark, chart, and README results verified.")
        sys.exit(0)


if __name__ == "__main__":
    main()
