"""
UCI Statlog (German Credit) dataset loader.

Classifies 1000 loan applicants as good or bad credit risks from 20
attributes (13 categorical, 7 numerical). Third benchmark dataset,
alongside COMPAS and Adult, for the Day 5 cross-dataset comparison.

Data source: the raw `german.data` file is not bundled with this repo
(same situation as Adult — see `data/README.md` for provenance).

Two sensitive attributes are derived here, following the convention used
in AIF360's own `GermanDataset` tutorial:
  - `sex_Male`, extracted from the combined `personal_status` field (the
    raw data encodes sex and marital status together).
  - `age_ge_25`, age binarized at 25 (AIF360's own default threshold for
    treating age as a protected attribute in this dataset).

IMPORTANT convention: like Adult and unlike COMPAS, label value 1 (good
credit risk) is the FAVORABLE outcome here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "german.data"

FAVORABLE_LABEL = 1
UNFAVORABLE_LABEL = 0

SENSITIVE_ATTRS = ["sex_Male", "age_ge_25"]
TARGET_COLUMNS = ["credit_risk"]

_RAW_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

# personal_status conflates sex and marital status in the raw data
# (A91/A93/A94 = male; A92/A95 = female). Decomposing this into a clean
# sex indicator is the whole reason it's handled specially rather than
# just one-hot encoded like the other categoricals below.
_MALE_CODES = {"A91", "A93", "A94"}

_CATEGORICAL_COLUMNS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
]


def load_german(
    path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str]]:
    """
    Load and preprocess the UCI German Credit dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, fully numeric (one-hot encoded categoricals).
    y_dict : dict[str, pd.Series]
        Mapping of target name -> label series. Single key ("credit_risk"),
        for interface consistency with the other loaders.
    sensitive_attrs : list[str]
        Column names in `X` usable as protected-attribute indicators.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_RAW_COLUMNS)

    # class: 1 = good credit risk, 2 = bad -> remap to 1/0 with 1 favorable.
    df["credit_risk"] = (df["class"] == 1).astype(int)
    df = df.drop(columns=["class"])

    df["sex_Male"] = df["personal_status"].isin(_MALE_CODES).astype(int)
    df = df.drop(columns=["personal_status"])

    df["age_ge_25"] = (df["age"] >= 25).astype(int)

    df = pd.get_dummies(df, columns=_CATEGORICAL_COLUMNS, drop_first=False, dtype=int)

    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    X = df[feature_cols]
    y_dict = {target: df[target] for target in TARGET_COLUMNS}

    sensitive_attrs = [c for c in SENSITIVE_ATTRS if c in X.columns]

    return X, y_dict, sensitive_attrs
