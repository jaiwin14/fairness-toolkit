"""
UCI Adult / Census Income dataset loader.

Predicts whether an individual's income exceeds $50K/yr from 1994 US
Census data. Standard fairness-literature benchmark alongside COMPAS.

Data source: the raw `adult.data` file is not bundled with this repo (see
`data/README.md`) — it's fetched from a public mirror once and cached
locally at `data/adult.data`. See that file's header for provenance.

IMPORTANT convention difference from `fairkit.datasets.compas`: here,
label value 1 (income >$50K) is the FAVORABLE outcome, unlike COMPAS where
0 (did not reoffend) is favorable. `FAVORABLE_LABEL` is exported so
downstream code (train/evaluate/mitigate) can stay dataset-agnostic by
reading it from the loader's module rather than assuming a fixed value.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "adult.data"

# Favorable/unfavorable label convention for THIS dataset (see module
# docstring — opposite of COMPAS's convention).
FAVORABLE_LABEL = 1
UNFAVORABLE_LABEL = 0

# Columns identified as protected/sensitive attributes after encoding.
SENSITIVE_ATTRS = ["sex_Male", "race_White"]

# Adult has a single target, unlike COMPAS's three — kept as a dict for a
# consistent load_*() -> (X, y_dict, sensitive_attrs) interface across
# dataset loaders.
TARGET_COLUMNS = ["income"]

_RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

# fnlwgt is a census sampling weight, not a real feature. education is
# redundant with education_num (same information, already ordinal).
# native_country is dropped: ~90% "United-States" with 40+ rare categories
# in the remainder, contributing little signal while ballooning column
# count under one-hot encoding.
_COLUMNS_TO_DROP = ["fnlwgt", "education", "native_country"]

_CATEGORICAL_COLUMNS = ["workclass", "marital_status", "occupation", "relationship", "race"]


def load_adult(
    path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str]]:
    """
    Load and preprocess the UCI Adult Income dataset.

    Parameters
    ----------
    path : str or Path
        Location of the raw `adult.data` file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, fully numeric (one-hot encoded categoricals).
    y_dict : dict[str, pd.Series]
        Mapping of target name -> label series. Only one key here
        ("income"), for interface consistency with `load_compas`.
    sensitive_attrs : list[str]
        Column names in `X` usable as protected-attribute indicators.
    """
    df = pd.read_csv(path, names=_RAW_COLUMNS, na_values="?", skipinitialspace=True)

    # Rows with missing workclass/occupation are a small minority (~5.6%)
    # and have no principled imputation here; drop rather than guess.
    df = df.dropna()

    df["income"] = (df["income"].str.strip().str.rstrip(".") == ">50K").astype(int)

    df = df.drop(columns=_COLUMNS_TO_DROP)
    df = pd.get_dummies(df, columns=_CATEGORICAL_COLUMNS, drop_first=False, dtype=int)
    df = pd.get_dummies(df, columns=["sex"], drop_first=True, dtype=int)  # -> sex_Male

    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    X = df[feature_cols]
    y_dict = {target: df[target] for target in TARGET_COLUMNS}

    sensitive_attrs = [c for c in SENSITIVE_ATTRS if c in X.columns]

    return X, y_dict, sensitive_attrs
