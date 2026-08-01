"""
Cross-dataset comparison: does bias mitigation help the accuracy/fairness
tradeoff consistently across datasets, or is it dataset-dependent?

Combines compas_results.csv, adult_results.csv, german_results.csv into
one table, and generates a grouped bar chart of disparate-impact
improvement per dataset per model for a single mitigation technique
(equalized_odds — the most consistently reliable technique across all
three benchmarks so far, per Days 3 and 4's findings).

Run with:  python scripts/compare_benchmarks.py
(run all three run_*_benchmark.py scripts first)

Writes:
  results/cross_dataset_comparison.csv
  results/cross_dataset_comparison.png
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = REPO_ROOT / "results"
MITIGATION_FOR_CHART = "equalized_odds"


def load_all_results() -> pd.DataFrame:
    frames = []
    for name in ["compas_results.csv", "adult_results.csv", "german_results.csv"]:
        df = pd.read_csv(RESULTS_DIR / name)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # COMPAS has two targets (is_recid, two_year_recid); the other two
    # datasets have one each. Keep only two_year_recid for COMPAS so every
    # dataset contributes exactly one row per (model, mitigation) — the
    # apples-to-apples comparison this script is for. (is_recid is still
    # in results/compas_results.csv in full, just not used here.)
    is_compas = combined["dataset_name"] == "compas"
    is_two_year = combined["target"] == "two_year_recid"
    combined = combined[~is_compas | is_two_year].copy()

    return combined


def bias_reduction_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dataset, model), how much did |DI ratio - 1| shrink from
    baseline to each mitigation technique? Positive = fairness improved.
    """
    df = df.copy()
    df["di_gap"] = (df["disparate_impact_ratio"] - 1.0).abs()

    baseline = df[df["mitigation"] == "none"].set_index(["dataset_name", "model_name"])["di_gap"]

    rows = []
    for _, row in df[df["mitigation"] != "none"].iterrows():
        key = (row["dataset_name"], row["model_name"])
        if key not in baseline.index:
            continue
        base_gap = baseline.loc[key]
        improvement = base_gap - row["di_gap"]
        rows.append({
            "dataset": row["dataset_name"],
            "model": row["model_name"],
            "mitigation": row["mitigation"],
            "baseline_di_gap": base_gap,
            "mitigated_di_gap": row["di_gap"],
            "di_gap_improvement": improvement,
            "accuracy_cost": baseline_accuracy(df, key) - row["accuracy"] if row["accuracy"] is not None else None,
        })
    return pd.DataFrame(rows)


def baseline_accuracy(df: pd.DataFrame, key: tuple) -> float:
    dataset_name, model_name = key
    match = df[
        (df["dataset_name"] == dataset_name)
        & (df["model_name"] == model_name)
        & (df["mitigation"] == "none")
    ]
    return match["accuracy"].iloc[0] if len(match) else float("nan")


def plot_comparison(table: pd.DataFrame):
    chart_df = table[table["mitigation"] == MITIGATION_FOR_CHART]
    datasets = sorted(chart_df["dataset"].unique())
    models = sorted(chart_df["model"].unique())

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"compas": "#1f77b4", "adult": "#ff7f0e", "german": "#2ca02c"}
    for i, dataset in enumerate(datasets):
        sub = chart_df[chart_df["dataset"] == dataset].set_index("model").reindex(models)
        ax.bar(
            x + (i - len(datasets) / 2 + 0.5) * width,
            sub["di_gap_improvement"],
            width,
            label=dataset,
            color=colors.get(dataset, None),
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Improvement in |disparate impact ratio - 1|\n(positive = more fair after mitigation)")
    ax.set_title(f"Cross-dataset fairness improvement from {MITIGATION_FOR_CHART}")
    ax.legend(title="dataset")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cross_dataset_comparison.png", dpi=150)
    print(f"Saved plot to {RESULTS_DIR / 'cross_dataset_comparison.png'}")


def main():
    combined = load_all_results()
    table = bias_reduction_table(combined)

    out_csv = RESULTS_DIR / "cross_dataset_comparison.csv"
    table.round(4).to_csv(out_csv, index=False)
    print(f"Saved {len(table)} rows to {out_csv}")

    plot_comparison(table)

    # Print a quick consistency summary to the console
    eo = table[table["mitigation"] == "equalized_odds"]
    print("\nequalized_odds fairness improvement (|DI-1| reduction) by dataset:")
    print(eo.groupby("dataset")["di_gap_improvement"].agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
