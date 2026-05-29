"""Model factory for the diamond price regression task."""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .data import RANDOM_STATE, inverse_log, log_transform
from .preprocessing import build_feature_preprocessor


def _wrap_log_target(estimator):
    """Wrap an estimator so it trains on log(price) but predicts in USD."""
    return TransformedTargetRegressor(
        regressor=estimator,
        func=log_transform,
        inverse_func=inverse_log,
        check_inverse=False,
    )


def build_mean_baseline() -> DummyRegressor:
    """Predicts the mean of training price (in original USD)."""
    return DummyRegressor(strategy="mean")


def build_linear_regression() -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_feature_preprocessor(scale_numeric=True)),
            ("model", LinearRegression()),
        ]
    )
    return _wrap_log_target(pipeline)


def build_random_forest() -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_feature_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return _wrap_log_target(pipeline)


def build_xgboost() -> TransformedTargetRegressor:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for this model. Install dependencies from "
            "requirements.txt before training all models."
        ) from exc

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_feature_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.1,
                    max_depth=6,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return _wrap_log_target(pipeline)


MODEL_BUILDERS = {
    "mean_baseline": build_mean_baseline,
    "linear_regression": build_linear_regression,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}
