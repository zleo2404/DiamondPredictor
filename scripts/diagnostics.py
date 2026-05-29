"""Cross-validated prediction diagnostics and feature-importance plots.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict

from diamond_price_prediction.data import RANDOM_STATE, load_and_prepare
from diamond_price_prediction.models import MODEL_BUILDERS

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    X_train, _, y_train, _ = load_and_prepare()
    y_true = y_train.values

    model_names = list(MODEL_BUILDERS.keys())
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    predictions = {
        name: cross_val_predict(builder(), X_train, y_train, cv=cv, n_jobs=1)
        for name, builder in MODEL_BUILDERS.items()
    }

    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 4))
    for ax, name in zip(axes, model_names):
        pred = predictions[name]
        ax.scatter(y_true, pred, s=4, alpha=0.25, color="#4477AA")
        lim = max(y_true.max(), pred.max())
        ax.plot([0, lim], [0, lim], "--", color="black", lw=1)
        ax.set_xlabel("actual (USD)")
        ax.set_ylabel("predicted (USD)")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnostics_predicted_vs_actual.png", dpi=130)
    plt.close(fig)

    informative = [n for n in model_names if n != "mean_baseline"]
    fig, axes = plt.subplots(1, len(informative), figsize=(4 * len(informative), 4))
    for ax, name in zip(axes, informative):
        pred = predictions[name]
        resid = y_true - pred
        ax.scatter(pred, resid, s=4, alpha=0.25, color="#EE6677")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("predicted (USD)")
        ax.set_ylabel("residual (USD)")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnostics_residuals.png", dpi=130)
    plt.close(fig)

    importance_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, ["random_forest", "xgboost"]):
        model = MODEL_BUILDERS[name]()
        model.fit(X_train, y_train)
        reg = model.regressor_ if hasattr(model, "regressor_") else model
        pre = reg.named_steps["preprocess"]
        feat_names = pre.get_feature_names_out()
        booster = reg.steps[-1][1]
        importances = booster.feature_importances_
        for feature, importance in zip(feat_names, importances):
            importance_rows.append(
                {
                    "model": name,
                    "feature": feature.split("__")[-1],
                    "importance": float(importance),
                }
            )
        order = np.argsort(importances)[::-1]
        sns.barplot(
            x=importances[order],
            y=[feat_names[i].split("__")[-1] for i in order],
            ax=ax,
            color="#4477AA",
        )
        ax.set_title(f"{name} — feature importance")
        ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnostics_feature_importance.png", dpi=130)
    plt.close(fig)

    pd.DataFrame(importance_rows).to_csv(
        TABLES_DIR / "feature_importance.csv", index=False, float_format="%.6f"
    )
    print(f"wrote diagnostic figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
