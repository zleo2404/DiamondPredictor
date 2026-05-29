"""Load the saved best-model checkpoint and run inference on a small sample.

Used to verify the persistence path end-to-end without evaluating the held-out
test split.

"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from diamond_price_prediction.data import load_and_prepare

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def main() -> None:
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    meta = json.loads((MODELS_DIR / "best_model.json").read_text())
    print(f"loaded model: {meta['name']}")
    print(
        f"selection metric: CV RMSE={meta['cv_rmse_mean']:.1f} "
        f"± {meta['cv_rmse_std']:.1f}"
    )

    X_train, _, y_train, _ = load_and_prepare()
    sample = X_train.head(10).copy()
    pred = model.predict(sample)
    out = pd.DataFrame(
        {
            "actual_usd": y_train.head(10).values,
            "predicted_usd": pred.round(2),
            "abs_error_usd": abs(y_train.head(10).values - pred).round(2),
        }
    )
    print("\nfirst 10 training-split predictions:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
