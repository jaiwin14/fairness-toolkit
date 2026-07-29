"""
COMPAS (ProPublica) dataset loader.

Encapsulates the preprocessing steps originally developed in
`notebooks/archive/Final_Processed_Propublica_Compas.ipynb` as clean,
callable, testable code.

Preprocessing summary:
    - Drop identifying / leakage-prone columns (name, dob, jail dates,
      recidivism-outcome-adjacent fields, redundant decile/priors columns).
    - One-hot encode `race` (all categories kept) and `sex` (drop_first,
      giving a single `sex_Male` indicator).
    - Bucket `age` into four groups (18-25, 26-35, 36-45, 46+) and
      one-hot encode them, dropping the raw `age` / `age_cat` columns.
    - One-hot encode `c_charge_degree`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default location of the raw CSV relative to the repo root.
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "compas-scores-two-years.csv"

# Columns identified as protected/sensitive attributes after encoding.
SENSITIVE_ATTRS = ["sex_Male", "race_African-American", "race_Caucasian"]

# Target columns available after preprocessing.
TARGET_COLUMNS = ["is_recid", "two_year_recid", "is_violent_recid"]

_AGE_BINS = [18, 25, 35, 45, float("inf")]
_AGE_LABELS = ["18-25", "26-35", "36-45", "46+"]

# Columns dropped because they are identifiers, free text, raw dates,
# post-outcome leakage, or duplicates of a column already retained.
_COLUMNS_TO_DROP = [
    "id", "name", "first", "last", "dob",
    "age", "age_cat",
    "compas_screening_date", "c_arrest_date", "c_days_from_compas",
    "c_case_number", "c_offense_date", "c_charge_desc",
    "c_jail_in", "c_jail_out",
    "r_case_number", "r_offense_date", "r_days_from_arrest",
    "r_charge_degree", "r_charge_desc", "r_jail_in", "r_jail_out",
    "violent_recid", "vr_case_number", "vr_offense_date",
    "vr_charge_degree", "vr_charge_desc",
    "type_of_assessment", "decile_score", "score_text", "decile_score.1",
    "v_type_of_assessment", "v_decile_score", "v_score_text", "v_screening_date",
    "days_b_screening_arrest", "priors_count.1",
    "in_custody", "out_custody", "start", "end", "event",
    "screening_date",
]


def _bucket_age(df: pd.DataFrame) -> pd.DataFrame:
    """Bin raw age into ordinal groups and one-hot encode them."""
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=True)
    df = pd.get_dummies(df, columns=["age_group"], prefix="age", dtype=int)
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode race, sex, and charge degree."""
    df = pd.get_dummies(df, columns=["race"], drop_first=False, dtype=int)
    df = pd.get_dummies(df, columns=["sex"], drop_first=True, dtype=int)
    df = pd.get_dummies(df, columns=["c_charge_degree"], drop_first=True, dtype=int)
    return df


def load_compas(
    path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str]]:
    """
    Load and preprocess the ProPublica COMPAS dataset.

    Parameters
    ----------
    path : str or Path
        Location of the raw `compas-scores-two-years.csv` file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, fully numeric (one-hot encoded categoricals).
    y_dict : dict[str, pd.Series]
        Mapping of target name -> label series, for each of
        `is_recid`, `two_year_recid`, `is_violent_recid`.
    sensitive_attrs : list[str]
        Column names in `X` usable as protected-attribute indicators
        for fairness evaluation (e.g. `sex_Male`, `race_African-American`).
    """
    df = pd.read_csv(path)

    df = _encode_categoricals(df)
    df = _bucket_age(df)

    df = df.drop(columns=_COLUMNS_TO_DROP)

    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    X = df[feature_cols]
    y_dict = {target: df[target] for target in TARGET_COLUMNS}

    sensitive_attrs = [c for c in SENSITIVE_ATTRS if c in X.columns]

    return X, y_dict, sensitive_attrs
