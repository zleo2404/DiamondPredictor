"""Loss / learning curves for each model.

- Linear Regression: learning curve (CV RMSE vs training-set size).
- Random Forest: CV RMSE vs n_estimators.
- XGBoost: per-boosting-round train / validation RMSE.
- MLP Regressor: per-iteration training loss and validation R2.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from diamond_price_prediction.data import (
    RANDOM_STATE,
    inverse_log,
    load_and_prepare,
    log_transform,
)
from diamond_price_prediction.models import (
    build_linear_regression,
    build_mlp_regressor,
)
from diamond_price_prediction.preprocessing import build_feature_preprocessor

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CV_FOLDS = 5
sns.set_theme(style="whitegrid", context="notebook")


def plot_linear_learning_curve(X_train, y_train):
    """Learning curve = CV score as a function of training-set size."""
    sizes = np.linspace(0.1, 1.0, 8)
    train_sizes, train_scores, val_scores = learning_curve(
        build_linear_regression(),
        X_train,
        y_train,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        train_sizes=sizes,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    train_rmse = np.sqrt(-train_scores).mean(axis=1)
    val_rmse = np.sqrt(-val_scores).mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_sizes, train_rmse, "o-", label="train", color="#4477AA")
    ax.plot(train_sizes, val_rmse, "o-", label="validation (5-fold CV)", color="#EE6677")
    ax.set_xlabel("training-set size")
    ax.set_ylabel("RMSE (USD)")
    ax.set_title("Linear Regression — learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_linear_learning_curve.png", dpi=130)
    plt.close(fig)
    print(f"  final train RMSE: {train_rmse[-1]:.0f}, val RMSE: {val_rmse[-1]:.0f}")


def plot_rf_n_estimators(X_train, y_train):
    """CV RMSE vs n_estimators."""
    n_values = [10, 25, 50, 100, 150, 200, 300]
    means, stds = [], []
    for n in n_values:
        model = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocess", build_feature_preprocessor(scale_numeric=False)),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=n, n_jobs=-1, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            func=log_transform,
            inverse_func=inverse_log,
            check_inverse=False,
        )
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_mean_squared_error",
            cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=1,  # rf is already parallel
        )
        rmse = np.sqrt(-scores)
        means.append(rmse.mean())
        stds.append(rmse.std())
        print(f"  n_estimators={n:3d}  CV RMSE={rmse.mean():.1f} ± {rmse.std():.1f}")

    means, stds = np.array(means), np.array(stds)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_values, means, "o-", color="#228833")
    ax.fill_between(n_values, means - stds, means + stds, alpha=0.2, color="#228833")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("CV RMSE (USD)")
    ax.set_title("Random Forest — CV RMSE vs n_estimators")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_rf_n_estimators.png", dpi=130)
    plt.close(fig)


def plot_xgb_boosting_curve(X_train, y_train):
    """Per-boosting-round train / validation RMSE (log-price scale)."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocess = build_feature_preprocessor(scale_numeric=False)
    X_tr_p = preprocess.fit_transform(X_tr)
    X_val_p = preprocess.transform(X_val)
    y_tr_log = np.log(y_tr.values)
    y_val_log = np.log(y_val.values)

    booster = XGBRegressor(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
        eval_metric="rmse",
    )
    booster.fit(
        X_tr_p,
        y_tr_log,
        eval_set=[(X_tr_p, y_tr_log), (X_val_p, y_val_log)],
        verbose=False,
    )
    history = booster.evals_result()
    train_rmse = history["validation_0"]["rmse"]
    val_rmse = history["validation_1"]["rmse"]
    rounds = range(1, len(train_rmse) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, train_rmse, label="train", color="#4477AA")
    ax.plot(rounds, val_rmse, label="validation", color="#EE6677")
    ax.set_xlabel("boosting round")
    ax.set_ylabel("RMSE (log-price)")
    ax.set_title("XGBoost — training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_xgb_boosting_curve.png", dpi=130)
    plt.close(fig)
    print(f"  final train RMSE (log): {train_rmse[-1]:.4f}, val RMSE (log): {val_rmse[-1]:.4f}")


def plot_mlp_training_curve(X_train, y_train):
    """Per-iteration MLP training loss and early-stopping validation R2."""
    model = build_mlp_regressor()
    model.fit(X_train, y_train)

    fitted_mlp = model.regressor_.named_steps["model"]
    loss_curve = fitted_mlp.loss_curve_
    validation_scores = getattr(fitted_mlp, "validation_scores_", None)
    has_validation_scores = (
        validation_scores is not None and len(validation_scores) > 0
    )

    if has_validation_scores:
        fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
        loss_ax, val_ax = axes
    else:
        fig, loss_ax = plt.subplots(figsize=(7, 4.5))
        val_ax = None

    iterations = range(1, len(loss_curve) + 1)
    loss_ax.plot(iterations, loss_curve, color="#4477AA")
    loss_ax.set_ylabel("Training loss on log-price")
    loss_ax.set_title("MLP Regressor — training curve")

    if val_ax is not None:
        val_iterations = range(1, len(validation_scores) + 1)
        val_ax.plot(val_iterations, validation_scores, color="#EE6677")
        val_ax.set_xlabel("iteration")
        val_ax.set_ylabel("Validation R2")
    else:
        loss_ax.set_xlabel("iteration")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_mlp_training_curve.png", dpi=130)
    plt.close(fig)

    print(f"  final training loss (log-price): {loss_curve[-1]:.4f}")
    if has_validation_scores:
        print(f"  final validation R2: {validation_scores[-1]:.4f}")


def main() -> None:
    X_train, _, y_train, _ = load_and_prepare()
    print(f"train: {X_train.shape}")

    print("\n--- Linear Regression learning curve ---")
    plot_linear_learning_curve(X_train, y_train)

    print("\n--- Random Forest CV RMSE vs n_estimators ---")
    plot_rf_n_estimators(X_train, y_train)

    print("\n--- XGBoost boosting curve ---")
    plot_xgb_boosting_curve(X_train, y_train)

    print("\n--- MLP Regressor training curve ---")
    plot_mlp_training_curve(X_train, y_train)

    print(f"\nWrote 4 loss/learning-curve figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
