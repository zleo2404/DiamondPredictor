"""Reusable preprocessing pipelines for the diamond regression task.

Design:
- `carat` is log1p-transformed first (right-skewed, dominant predictor).
- For linear models, all numerics are then standard-scaled.
- For tree-based models, scaling is skipped (does not affect tree splits).
- Ordinal categoricals (cut, color, clarity) are already integer-encoded in
  `data.encode_ordinals` and pass through unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

OTHER_NUMERIC = [f for f in NUMERIC_FEATURES if f != "carat"]


def build_feature_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Column transformer for diamond features.

    Parameters
    ----------
    scale_numeric : bool
        True for models sensitive to feature scale (Linear Regression).
        False for tree-based models (Random Forest, XGBoost).
    """
    log_pipeline = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

    if scale_numeric:
        carat_pipeline = Pipeline(
            steps=[("log", log_pipeline), ("scale", StandardScaler())]
        )
        other_numeric_pipeline = StandardScaler()
    else:
        carat_pipeline = log_pipeline
        other_numeric_pipeline = "passthrough"

    return ColumnTransformer(
        transformers=[
            ("carat", carat_pipeline, ["carat"]),
            ("num", other_numeric_pipeline, OTHER_NUMERIC),
            ("cat", "passthrough", CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
