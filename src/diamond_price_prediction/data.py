"""Data loading and train/test splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "diamonds.csv"

NUMERIC_FEATURES = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL_FEATURES = ["cut", "color", "clarity"]
TARGET = "price"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]

#Domain of ordinal rankings. Higher = better quality.
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

RANDOM_STATE = 42


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw diamonds CSV, dropping any unnamed index column."""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    validate_raw(df)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    """Validate the raw dataset schema and basic target constraints."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("raw dataset contains missing values")

    if (df[TARGET] <= 0).any():
        raise ValueError("target column must contain strictly positive prices")


