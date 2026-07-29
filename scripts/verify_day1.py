"""
Day 1 trial run — verifies that the Day 1 checkpoint is genuinely met.

Run with:  python scripts/verify_day1.py

Checks:
    1. Expected repo folders/files exist.
    2. requirements.txt is populated.
    3. load_compas() runs standalone (no notebook needed).
    4. Output matches the shape/rates we validated against the original
       notebook (7214 rows, ~45.1% two_year_recid rate).
    5. No NaNs, sensitive attributes present, targets binary.

Exits with code 0 and a "DAY 1 CHECKPOINT: PASSED" message if everything
holds, or exits 1 and prints which check failed.
"""

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
    print("DAY 1 TRIAL RUN")
    print("=" * 60)

    # 1. Repo structure
    expected_paths = [
        "src/fairkit/__init__.py",
        "src/fairkit/datasets/compas.py",
        "tests/test_compas_loader.py",
        "notebooks/archive",
        "results",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
    ]
    for rel in expected_paths:
        check(f"path exists: {rel}", (REPO_ROOT / rel).exists())

    # 2. requirements.txt populated
    req_file = REPO_ROOT / "requirements.txt"
    if req_file.exists():
        n_lines = len([l for l in req_file.read_text().splitlines() if l.strip()])
        check("requirements.txt has pinned packages", n_lines > 10, f"found {n_lines} lines")

    # 3 + 4 + 5. load_compas() actually runs
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from fairkit.datasets.compas import load_compas

        X, y_dict, sensitive_attrs = load_compas()

        check("load_compas() runs without error", True)
        check("X is non-empty", X.shape[0] > 0 and X.shape[1] > 0, f"shape={X.shape}")
        check("row count matches original notebook (7214)", X.shape[0] == 7214, f"got {X.shape[0]}")
        check("no NaNs in features", not X.isnull().any().any())
        check("sensitive attributes present", len(sensitive_attrs) > 0, f"got {sensitive_attrs}")
        check("sex_Male is a sensitive attribute", "sex_Male" in sensitive_attrs)

        rate = y_dict["two_year_recid"].mean()
        check(
            "two_year_recid rate matches notebook (~45.1%)",
            abs(rate - 0.451) < 0.005,
            f"got {rate:.3f}",
        )

        for target, series in y_dict.items():
            values = set(series.dropna().unique().tolist())
            check(f"{target} is binary", values.issubset({0, 1}), f"got {values}")

    except Exception as e:
        check("load_compas() runs without error", False, str(e))

    print("=" * 60)
    if failures:
        print(f"DAY 1 CHECKPOINT: FAILED ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("DAY 1 CHECKPOINT: PASSED — repo skeleton + load_compas() verified end to end.")
        sys.exit(0)


if __name__ == "__main__":
    main()
