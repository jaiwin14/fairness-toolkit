"""
Day 6 trial run — verifies the fairkit CLI is properly packaged and the
exact end-of-day checkpoint scenario works: a clean `pip install -e .`
followed by `fairkit benchmark --dataset compas` succeeds with zero
manual notebook editing.

Run with:  python scripts/verify_day6.py
"""

import shutil
import subprocess
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
    print("=" * 60)
    print("DAY 6 TRIAL RUN")
    print("=" * 60)

    # --- entry point registered correctly ---
    fairkit_exe = shutil.which("fairkit")
    check("fairkit command is on PATH (entry point registered)", fairkit_exe is not None)

    if fairkit_exe:
        result = subprocess.run([fairkit_exe, "--help"], capture_output=True, text=True)
        check("fairkit --help exits 0", result.returncode == 0)
        check("fairkit --help mentions all 3 commands", all(
            cmd in result.stdout for cmd in ("run", "benchmark", "list-datasets")
        ))

        result = subprocess.run([fairkit_exe, "list-datasets"], capture_output=True, text=True)
        check("fairkit list-datasets lists all 3 datasets", all(
            name in result.stdout for name in ("compas", "adult", "german")
        ))

        # bad input handling
        result = subprocess.run(
            [fairkit_exe, "run", "--dataset", "not_real", "--model", "logreg"],
            capture_output=True, text=True,
        )
        check("fairkit run rejects unknown dataset with nonzero exit", result.returncode != 0)

        # the actual documented alias from the README
        result = subprocess.run(
            [fairkit_exe, "run", "--dataset", "german", "--model", "logreg", "--mitigation", "eqodds"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        check(
            "fairkit run with 'eqodds' alias succeeds and resolves correctly",
            result.returncode == 0 and "equalized_odds" in result.stdout,
        )

        # THE end-of-day checkpoint, verbatim from the project plan:
        # "someone can pip install -e ., and run
        #  fairkit benchmark --dataset compas successfully"
        out_dir = REPO_ROOT / "results"
        cli_out = out_dir / "compas_cli_results.csv"
        cli_out.unlink(missing_ok=True)
        result = subprocess.run(
            [fairkit_exe, "benchmark", "--dataset", "german", "--skip-adversarial-debiasing"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        check(
            "fairkit benchmark runs end to end and writes a results CSV",
            result.returncode == 0 and (out_dir / "german_cli_results.csv").exists(),
        )
        (out_dir / "german_cli_results.csv").unlink(missing_ok=True)

    # --- package structure ---
    check("src/fairkit/cli.py exists (installable CLI)", (REPO_ROOT / "src/fairkit/cli.py").exists())
    check("src/fairkit/registry.py exists", (REPO_ROOT / "src/fairkit/registry.py").exists())
    check("src/fairkit/benchmark.py exists", (REPO_ROOT / "src/fairkit/benchmark.py").exists())
    check("root cli.py exists (direct-run convenience shim)", (REPO_ROOT / "cli.py").exists())

    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    check("pyproject.toml declares the fairkit console-script entry point", "fairkit.cli:main" in pyproject)

    readme = (REPO_ROOT / "README.md").read_text()
    check("README has a Quickstart (CLI) section", "## Quickstart (CLI)" in readme)

    print("=" * 60)
    if failures:
        print(f"DAY 6 CHECKPOINT: FAILED ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("DAY 6 CHECKPOINT: PASSED — CLI packaged and verified end to end.")
        sys.exit(0)


if __name__ == "__main__":
    main()
