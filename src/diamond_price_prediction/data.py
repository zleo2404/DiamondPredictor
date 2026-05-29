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

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically invalid rows, duplicates, and extreme dimensional 
    outliers(where y or z exceed 30 mm since the largest legitimate diamond in the set has y < 12 mm)).
    """
    df = df.loc[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
    df = df.loc[(df["y"] < 30) & (df["z"] < 30)].copy()
    return df.drop_duplicates().reset_index(drop=True)


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map cut, color, clarity to integer ranks reflecting their natural order."""
    df = df.copy()
    df["cut"] = df["cut"].map({v: i for i, v in enumerate(CUT_ORDER)})
    df["color"] = df["color"].map({v: i for i, v in enumerate(COLOR_ORDER)})
    df["clarity"] = df["clarity"].map({v: i for i, v in enumerate(CLARITY_ORDER)})

    if df[CATEGORICAL_FEATURES].isna().any().any():
        raise ValueError("categorical columns contain values outside the defined order")

    return df

def split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split on carat quintiles.
    
    Returns: X_train, X_test, y_train, y_test (price in original USD).
    """
    layers = pd.qcut(df["carat"], q=5, labels=False, duplicates="drop")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=layers,
    )


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline to validate the dataset"""
    df = load_raw()
    df = clean(df)
    df = encode_ordinals(df)
    return split(df)


def log_transform(y: pd.Series | np.ndarray) -> np.ndarray:
    """Natural log of price. All prices in the dataset are strictly positive."""
    return np.log(y)


def inverse_log(y: np.ndarray) -> np.ndarray:
    """Inverse of log transform."""
    return np.exp(y)



