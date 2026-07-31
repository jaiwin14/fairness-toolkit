"""
Generate the accuracy-vs-fairness tradeoff chart from results/compas_results.csv.

Run with:  python scripts/plot_compas_tradeoff.py
(run scripts/run_compas_benchmark.py first if compas_results.csv doesn't exist yet)

Writes results/compas_tradeoff.png.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # no display needed, just write a file
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = REPO_ROOT / "results" / "compas_results.csv"
OUT_PNG = REPO_ROOT / "results" / "compas_tradeoff.png"

# Colors for mitigation techniques (shared across markers below)
MITIGATION_COLORS = {
    "none": "#444444",
    "reject_option_classification": "#d62728",
    "equalized_odds": "#2ca02c",
    "calibrated_equalized_odds": "#9467bd",
    "adversarial_debiasing": "#1f77b4",
}
MODEL_MARKERS = {
    "logreg": "o",
    "svc": "s",
    "gbc": "^",
    "xgboost": "D",
    "adversarial_debiasing_nn": "*",
}


def plot_target(ax, df: pd.DataFrame, target: str):
    subset = df[df["target"] == target]
    for _, row in subset.iterrows():
        color = MITIGATION_COLORS.get(row["mitigation"], "#999999")
        marker = MODEL_MARKERS.get(row["model_name"], "x")
        size = 220 if row["mitigation"] == "none" else 130
        ax.scatter(
            row["disparate_impact_ratio"], row["accuracy"],
            c=color, marker=marker, s=size,
            edgecolors="black" if row["mitigation"] == "none" else "none",
            linewidths=1.2, alpha=0.9, zorder=3,
        )
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
    ax.set_xlabel("Disparate impact ratio  (1.0 = fair)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"target: {target}")
    ax.grid(alpha=0.25, zorder=0)


def main():
    df = pd.read_csv(RESULTS_CSV)
    # compas_results.csv doesn't have a "target" column yet — the benchmark
    # script runs one target at a time, so infer it from row order/DP diff
    # patterns is unreliable; instead, require the column and fail loudly
    # if the CSV predates it, so this never silently plots garbage.
    if "target" not in df.columns:
        raise SystemExit(
            "results/compas_results.csv has no 'target' column — "
            "re-run scripts/run_compas_benchmark.py to regenerate it."
        )

    df = df.dropna(subset=["accuracy", "disparate_impact_ratio"])
    # cap extreme/infinite DI ratios for plotting only (raw CSV keeps true values)
    df["disparate_impact_ratio"] = df["disparate_impact_ratio"].clip(upper=3.0)

    targets = sorted(df["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 6), squeeze=False)
    axes = axes[0]

    for ax, target in zip(axes, targets):
        plot_target(ax, df, target)

    # shared legend built manually (mitigation = color, model = marker shape)
    mitigation_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=m)
        for m, c in MITIGATION_COLORS.items()
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=mk, color="black", linestyle="None", markersize=9, label=name)
        for name, mk in MODEL_MARKERS.items()
    ]
    fig.legend(
        handles=mitigation_handles + model_handles,
        loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08),
        fontsize=9, frameon=False,
    )
    fig.suptitle("COMPAS: accuracy vs. disparate impact, pre/post mitigation", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
