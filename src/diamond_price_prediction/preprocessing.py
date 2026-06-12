"""Reusable preprocessing pipelines for the diamond regression task.

Design:
- `carat` is log1p-transformed first (right-skewed, dominant predictor).
- For linear models, all numerics are then standard-scaled.
- For tree-based models, scaling is skipped (does not affect tree splits).
- Ordinal categoricals (cut, color, clarity) are already integer-encoded in
  `data.encode_ordinals`; they can pass through unchanged or be one-hot encoded.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    CLARITY_ORDER,
    COLOR_ORDER,
    CUT_ORDER,
    NUMERIC_FEATURES,
)

OTHER_NUMERIC = [f for f in NUMERIC_FEATURES if f != "carat"]
CATEGORY_RANGES = [
    range(len(CUT_ORDER)),
    range(len(COLOR_ORDER)),
    range(len(CLARITY_ORDER)),
]


def build_feature_preprocessor(
    scale_numeric: bool,
    categorical: str = "ordinal",
) -> ColumnTransformer:
    """Column transformer for diamond features.

    Parameters
    ----------
    scale_numeric : bool
        True for models sensitive to feature scale (Linear Regression).
        False for tree-based models (Random Forest, XGBoost).
    categorical : {"ordinal", "onehot"}
        How to represent already ordinal-encoded categorical columns.
    """
    if categorical not in {"ordinal", "onehot"}:
        raise ValueError(
            "categorical must be either 'ordinal' or 'onehot', "
            f"got {categorical!r}"
        )

    log_pipeline = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

    if scale_numeric:
        carat_pipeline = Pipeline(
            steps=[("log", log_pipeline), ("scale", StandardScaler())]
        )
        other_numeric_pipeline = StandardScaler()
    else:
        carat_pipeline = log_pipeline
        other_numeric_pipeline = "passthrough"

    if categorical == "ordinal":
        categorical_pipeline = "passthrough"
    else:
        categorical_pipeline = OneHotEncoder(
            categories=CATEGORY_RANGES,
            sparse_output=False,
            handle_unknown="error",
        )

    return ColumnTransformer(
        transformers=[
            ("carat", carat_pipeline, ["carat"]),
            ("num", other_numeric_pipeline, OTHER_NUMERIC),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
