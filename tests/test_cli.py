import warnings

import pytest
from click.testing import CliRunner

from fairkit.cli import cli

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


def test_help_exits_zero(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "fairkit" in result.output.lower()


def test_list_datasets(runner):
    result = runner.invoke(cli, ["list-datasets"])
    assert result.exit_code == 0
    for name in ("compas", "adult", "german"):
        assert name in result.output


def test_list_models(runner):
    result = runner.invoke(cli, ["list-models"])
    assert result.exit_code == 0
    for name in ("logreg", "svc", "gbc", "xgboost"):
        assert name in result.output


def test_run_rejects_unknown_dataset(runner):
    result = runner.invoke(cli, ["run", "--dataset", "not_a_dataset", "--model", "logreg"])
    assert result.exit_code != 0
    assert "not one of" in result.output


def test_run_rejects_unknown_model(runner):
    result = runner.invoke(cli, ["run", "--dataset", "compas", "--model", "not_a_model"])
    assert result.exit_code != 0


def test_run_rejects_unknown_mitigation(runner):
    result = runner.invoke(
        cli, ["run", "--dataset", "compas", "--model", "logreg", "--mitigation", "not_a_technique"]
    )
    assert result.exit_code != 0


def test_run_missing_required_option_fails(runner):
    result = runner.invoke(cli, ["run", "--dataset", "compas"])
    assert result.exit_code != 0


def test_run_baseline_on_smallest_dataset(runner):
    # german is the smallest dataset (1000 rows) -- fast enough for a real test.
    result = runner.invoke(cli, ["run", "--dataset", "german", "--model", "logreg"])
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.output
    assert "disparate_impact_ratio" in result.output


def test_run_with_mitigation_alias(runner):
    # "eqodds" is documented as a valid alias for equalized_odds.
    result = runner.invoke(
        cli, ["run", "--dataset", "german", "--model", "logreg", "--mitigation", "eqodds"]
    )
    assert result.exit_code == 0, result.output
    assert "mitigation: equalized_odds" in result.output


def test_benchmark_writes_csv(runner, tmp_path):
    result = runner.invoke(
        cli,
        [
            "benchmark", "--dataset", "german",
            "--output", str(tmp_path), "--skip-adversarial-debiasing",
        ],
    )
    assert result.exit_code == 0, result.output
    out_file = tmp_path / "german_cli_results.csv"
    assert out_file.exists()
    assert "Saved" in result.output
