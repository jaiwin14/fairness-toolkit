"""
fairkit CLI.

Thin glue over `fairkit.benchmark`, `fairkit.registry`, and
`fairkit.mitigate` — all the real work happened Days 1-5; this just
exposes it as commands.

Examples
--------
    fairkit run --dataset compas --model xgboost --mitigation eqodds
    fairkit benchmark --dataset compas --output results/
    fairkit benchmark --dataset all --output results/
    fairkit list-datasets
    fairkit list-models
"""

from __future__ import annotations

from pathlib import Path

import click

from .benchmark import run_benchmark, run_single
from .mitigate import ALL_TECHNIQUES
from .registry import DATASET_REGISTRY
from .train import AVAILABLE_MODELS

# Short aliases for mitigation names, so `--mitigation eqodds` (as sketched
# in the original project plan) works alongside the full name.
MITIGATION_ALIASES = {
    "none": "none",
    "roc": "reject_option_classification",
    "eqodds": "equalized_odds",
    "ceo": "calibrated_equalized_odds",
    "adv": "adversarial_debiasing",
}
MITIGATION_CHOICES = sorted(set(MITIGATION_ALIASES) | {"none"} | set(ALL_TECHNIQUES))


def _resolve_mitigation(value: str) -> str:
    return MITIGATION_ALIASES.get(value, value)


@click.group()
@click.version_option(package_name="fairkit")
def cli():
    """fairkit — bias mitigation benchmarking across recidivism/credit/income datasets."""


@cli.command()
@click.option(
    "--dataset", required=True, type=click.Choice(sorted(DATASET_REGISTRY)),
    help="Which dataset to use.",
)
@click.option(
    "--model", required=True, type=click.Choice(sorted(AVAILABLE_MODELS)),
    help="Which model to train.",
)
@click.option(
    "--mitigation", default="none", type=click.Choice(MITIGATION_CHOICES),
    help="Bias mitigation technique to apply (or 'none' for baseline). "
         "Aliases: roc, eqodds, ceo, adv.",
)
def run(dataset: str, model: str, mitigation: str):
    """Train one model on one dataset, optionally apply one mitigation technique, print results."""
    resolved = _resolve_mitigation(mitigation)
    click.echo(f"Running {model} on {dataset} (mitigation: {resolved})...")
    result = run_single(dataset_name=dataset, model_name=model, mitigation=resolved)

    click.echo("")
    click.echo(f"  accuracy                      = {result['accuracy']:.4f}")
    click.echo(f"  demographic_parity_difference = {result['demographic_parity_difference']:.4f}")
    click.echo(f"  equalized_odds_difference     = {result['equalized_odds_difference']:.4f}")
    click.echo(f"  disparate_impact_ratio        = {result['disparate_impact_ratio']:.4f}  (1.0 = fair)")


@cli.command()
@click.option(
    "--dataset", required=True,
    type=click.Choice(sorted(DATASET_REGISTRY) + ["all"]),
    help="Which dataset to benchmark, or 'all' for every dataset.",
)
@click.option(
    "--output", default="results", type=click.Path(file_okay=False),
    help="Directory to write <dataset>_cli_results.csv into. Default: results/",
)
@click.option(
    "--skip-adversarial-debiasing", is_flag=True, default=False,
    help="Skip the (slower) adversarial debiasing in-processing technique.",
)
def benchmark(dataset: str, output: str, skip_adversarial_debiasing: bool):
    """Run every model x every mitigation technique on one dataset (or all of them)."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = sorted(DATASET_REGISTRY) if dataset == "all" else [dataset]

    for name in targets:
        click.echo(f"=== dataset: {name} ===")
        df = run_benchmark(name, include_adversarial_debiasing=not skip_adversarial_debiasing)
        out_path = out_dir / f"{name}_cli_results.csv"
        df.to_csv(out_path, index=False)
        click.echo(f"Saved {len(df)} rows to {out_path}\n")


@cli.command("list-datasets")
def list_datasets():
    """List available dataset names."""
    for name, spec in sorted(DATASET_REGISTRY.items()):
        click.echo(f"  {name:8s}  target={spec.default_target}")


@cli.command("list-models")
def list_models():
    """List available model names."""
    for name in sorted(AVAILABLE_MODELS):
        click.echo(f"  {name}")


def main():
    cli()


if __name__ == "__main__":
    main()
