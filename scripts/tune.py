from __future__ import annotations

import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from diamond_price_prediction.data import load_and_prepare, RANDOM_STATE
from diamond_price_prediction.models import MODEL_BUILDERS

def main() -> None:
    print("=== Starting Hyperparameter Tuning Pipeline ===")
    
    splits = load_and_prepare()
    X_train, y_train = splits[0], splits[2]
    print(f"Training data loaded: {X_train.shape}\n")

    tuning_grids = {
        "xgboost": {
            "regressor__model__n_estimators": [200, 400],
            "regressor__model__learning_rate": [0.05, 0.1],
            "regressor__model__max_depth": [4, 6, 8],
            "regressor__model__subsample": [0.8, 1.0]
        },
        "random_forest": {
            "regressor__model__n_estimators": [200, 300],
            "regressor__model__max_depth": [None, 15, 20],
            "regressor__model__min_samples_split": [2, 5]
        },
        "mlp_regressor": {  
            "regressor__model__hidden_layer_sizes": [(64, 32), (128, 64)],
            "regressor__model__learning_rate_init": [0.001, 0.005],
            "regressor__model__alpha": [1e-3, 1e-2]
        }
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for model_name, param_grid in tuning_grids.items():
        if model_name not in MODEL_BUILDERS:
            print(f"Warning: Model '{model_name}' not found in MODEL_BUILDERS. Skipping...")
            continue
            
        print(f"Starting tuning for: {model_name}...")
        print(f"Search space: {len(param_grid)} base hyperparameters.")
        
        base_model = MODEL_BUILDERS[model_name]()
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        
        t0 = time.time()
        grid_search.fit(X_train, y_train)
        dt = time.time() - t0
        
        best_params_clean = {
            k.replace("regressor__model__", ""): v 
            for k, v in grid_search.best_params_.items()
        }
        
        print("\n" + "="*50)
        print(f"TUNING COMPLETED: {model_name.upper()} (in {dt:.1f} seconds)")
        print(f"Best RMSE (3-Fold CV): {-grid_search.best_score_:.2f} USD")
        print("PARAMETERS TO UPDATE IN models.py:")
        for param, value in best_params_clean.items():
            print(f"    {param} = {value}")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()