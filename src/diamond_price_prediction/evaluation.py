"""Evaluation metrics and reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return RMSE, MAE, R², MAPE on the original USD scale."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {"rmse_usd": rmse, "mae_usd": mae, "r2": r2, "mape": mape}


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Format a list of metric dicts into a comparison DataFrame."""
    return pd.DataFrame(rows).set_index("model")
