"""
Registry of dataset loaders and their conventions.

Each dataset module (`fairkit.datasets.compas`, `.adult`, `.german`) has
its own favorable-label / privileged-value conventions (see Day 4's
generalization work in `fairkit.mitigate`). This module is the single
place those conventions get looked up by name, so the CLI and benchmark
runner don't hardcode dataset-specific branching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .datasets import adult, compas, german


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable[[], tuple[pd.DataFrame, dict[str, pd.Series], list[str]]]
    default_target: str
    favorable_label: int
    unfavorable_label: int
    privileged_value: int


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "compas": DatasetSpec(
        name="compas",
        loader=compas.load_compas,
        default_target="two_year_recid",
        favorable_label=0,
        unfavorable_label=1,
        privileged_value=0,
    ),
    "adult": DatasetSpec(
        name="adult",
        loader=adult.load_adult,
        default_target="income",
        favorable_label=adult.FAVORABLE_LABEL,
        unfavorable_label=adult.UNFAVORABLE_LABEL,
        privileged_value=1,
    ),
    "german": DatasetSpec(
        name="german",
        loader=german.load_german,
        default_target="credit_risk",
        favorable_label=german.FAVORABLE_LABEL,
        unfavorable_label=german.UNFAVORABLE_LABEL,
        privileged_value=1,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]
