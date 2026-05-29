"""Train candidate models with cross-validation and persist checkpoints.

Outputs:
    reports/tables/metrics.csv           - cross-validation metrics per model
    reports/figures/metrics_bar.png      - cross-validation metric chart
    models/best_model.joblib             - selected model checkpoint
    models/best_model.json               - selected model metadata

Run from project root:
    PYTHONPATH=src python scripts/train.py

Final held-out evaluation is intentionally opt-in:
    PYTHONPATH=src python scripts/train.py --final-eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate

from diamond_price_prediction.data import load_and_prepare
from diamond_price_prediction.evaluation import regression_metrics
from diamond_price_prediction.models import MODEL_BUILDERS

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "reports" / "tables"
FIGS_DIR = ROOT / "reports" / "figures"
for d in (MODELS_DIR, TABLES_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CV_FOLDS = 5
RANDOM_STATE = 42
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
    "mape": "neg_mean_absolute_percentage_error",
}

sns.set_theme(style="whitegrid", context="notebook")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="evaluate fitted models on the held-out test set",
    )
    return parser.parse_args()


def summarize_cv_scores(scores: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "cv_rmse_mean": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_mae_std": float(scores["test_mae"].std()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
        "cv_mape_mean": float(-scores["test_mape"].mean()),
        "cv_mape_std": float(scores["test_mape"].std()),
    }


def save_cv_metric_chart(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    chart_specs = [
        ("cv_rmse_mean", "CV RMSE (USD)", True),
        ("cv_mae_mean", "CV MAE (USD)", True),
        ("cv_r2_mean", "CV R2", False),
    ]
    for ax, (metric, title, ascending) in zip(axes, chart_specs):
        order = table[metric].sort_values(ascending=ascending).index
        sns.barplot(
            x=table.loc[order].index,
            y=table.loc[order, metric],
            ax=ax,
            color="#4477AA",
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "metrics_bar.png", dpi=130)
    plt.close(fig)


def evaluate_held_out(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append({"model": name, **regression_metrics(y_test, model.predict(X_test))})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(TABLES_DIR / "final_metrics.csv", float_format="%.4f")
    return table


def main() -> None:
    args = parse_args()
    splits = load_and_prepare()
    X_train, y_train = splits[0], splits[2]
    print(f"train: {X_train.shape}")
    if not args.final_eval:
        print("held-out test evaluation deferred; use --final-eval for final phase")
    print()

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    fitted_models = {}

    for name, builder in MODEL_BUILDERS.items():
        print(f"=== {name} ===")
        model = builder()

        t0 = time.time()
        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=SCORING,
            cv=kf,
            n_jobs=1,
        )
        cv_metrics = summarize_cv_scores(scores)
        cv_dt = time.time() - t0
        print(
            f"  CV RMSE (USD): {cv_metrics['cv_rmse_mean']:.1f}"
            f" ± {cv_metrics['cv_rmse_std']:.1f}"
            f"  ({cv_dt:.1f}s)"
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        fit_dt = time.time() - t0
        print(f"  fit on full training split: {fit_dt:.1f}s")

        fitted_models[name] = model

        rows.append(
            {
                "model": name,
                **cv_metrics,
                "fit_seconds": fit_dt,
            }
        )

    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(TABLES_DIR / "metrics.csv", float_format="%.4f")
    print("\n=== summary ===")
    print(table.round(4))

    save_cv_metric_chart(table)

    best_name = table["cv_rmse_mean"].idxmin()
    best_dst = MODELS_DIR / "best_model.joblib"
    joblib.dump(fitted_models[best_name], best_dst, compress=3)

    metadata = {
        "name": best_name,
        "selection_metric": "cv_rmse_mean",
        **table.loc[best_name].to_dict(),
    }

    if args.final_eval:
        X_test, y_test = splits[1], splits[3]
        print(f"\nheld-out test: {X_test.shape}")
        final_table = evaluate_held_out(fitted_models, X_test, y_test)
        print("\n=== held-out final evaluation ===")
        print(final_table.round(4))
        metadata["held_out_metrics"] = final_table.loc[best_name].to_dict()

    (MODELS_DIR / "best_model.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nBest model: {best_name}  (saved to {best_dst})")


if __name__ == "__main__":
    main()
