"""
Day 7 trial run — verifies the final-polish checklist: README tells the
full story, citations/license/gitignore are in order, the technical
summary exists, and the whole repo still works end to end.

Run with:  python scripts/verify_day7.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

failures = []


def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def main():
    print("=" * 60)
    print("DAY 7 TRIAL RUN")
    print("=" * 60)

    readme = (REPO_ROOT / "README.md").read_text()

    # --- README tells the full story ---
    check("README has a headline finding up top", "## The headline finding" in readme)
    check("README has a Quickstart section", "## Quickstart" in readme)
    check("README has a Methodology section", "## Methodology" in readme)
    check("README has a Cross-dataset comparison section", "## Cross-dataset comparison" in readme)
    check("README has collapsible per-dataset detail (skimmable)", "<details>" in readme)
    check("README cites all 3 data sources", all(
        s in readme for s in ("ProPublica", "Becker, B. & Kohavi, R.", "Hofmann, H.")
    ))
    check("README links to LICENSE", "[LICENSE](LICENSE)" in readme)
    check("README no longer says 'Work in progress' placeholder", "Work in progress" not in readme)

    # README length sanity: should be readable in ~2 minutes at a skim.
    # A skim reads roughly the non-table, non-collapsed content; can't
    # measure "2 minutes" directly, so check total line count is in a
    # reasonable range for a thorough-but-skimmable doc (not a 1000+ line
    # wall of unstructured text).
    n_lines = len(readme.splitlines())
    check("README length is reasonable (not an unstructured wall of text)", n_lines < 400, f"{n_lines} lines")

    # --- final cleanup ---
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    check(
        "gitignore no longer excludes results/ (recruiters should see them)",
        "results/*.csv" not in gitignore and "results/*.png" not in gitignore,
    )

    check("docs/project_summary.md exists", (REPO_ROOT / "docs/project_summary.md").exists())
    if (REPO_ROOT / "docs/project_summary.md").exists():
        summary_words = len((REPO_ROOT / "docs/project_summary.md").read_text().split())
        check(
            "project_summary.md is roughly 300-400 words",
            250 <= summary_words <= 450,
            f"got {summary_words} words",
        )

    # --- the whole thing still works ---
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    check("full test suite passes (45 tests)", result.returncode == 0 and "45 passed" in result.stdout)

    print("=" * 60)
    if failures:
        print(f"DAY 7 CHECKPOINT: FAILED ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("DAY 7 CHECKPOINT: PASSED — final polish verified. Repo is ready to publish.")
        sys.exit(0)


if __name__ == "__main__":
    main()
