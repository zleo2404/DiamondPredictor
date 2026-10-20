# Diamond Price Prediction: A Comparative Study of Regression Models

A reproducible machine learning pipeline that predicts diamond prices from physical measurements and quality grades, and compares model families with different inductive biases: linear regression, Random Forest, gradient boosting (XGBoost) and a neural network.

Developed as a team project for the EARIN course at Warsaw University of Technology (Group 16).

## Overview

The task is supervised regression on the standard `diamonds` dataset (53,940 records). The target is `price` in US dollars; the input features are `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y` and `z`.

The project goes beyond fitting a single predictor. Its goal is a fair, leakage-free comparison of models against a mean-price baseline, using cross-validation for model selection and a single held-out evaluation for the final result.

## Results

Final evaluation on the held-out test set (10,755 diamonds), with errors reported in USD:

| Model | RMSE | MAE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Mean baseline | 4042.03 | 3057.65 | -0.0001 | 187.7% |
| Linear regression | 1420.77 | 567.83 | 0.8764 | 12.7% |
| Random Forest | 543.06 | 271.14 | 0.9819 | 6.4% |
| **XGBoost** | **534.80** | **269.49** | **0.9825** | **6.3%** |
| MLP regressor | 564.25 | 291.74 | 0.9805 | 7.4% |

XGBoost was selected as the best model by five-fold cross-validated RMSE on the training split (524.50 ± 8.22 USD). Tree ensembles capture the nonlinear interactions between size and quality grades substantially better than the linear baseline, while the MLP is competitive but does not improve on boosting or Random Forest.

## Engineering Highlights

- **Modular, reusable package.** Data loading, preprocessing, model construction and evaluation live in a dedicated package (`src/diamond_price_prediction`), while scripts only orchestrate experiments.
- **Data validation and cleaning.** The schema, missing values and target constraints are validated at load time. Physically invalid records, extreme dimensional outliers and duplicates are removed, reducing the data from 53,940 to 53,772 rows.
- **Leakage-safe pipelines.** Preprocessing is part of each scikit-learn `Pipeline`, so it is fitted only on training folds during cross-validation.
- **Model-aware preprocessing.** Feature scaling is applied only to scale-sensitive models; ordinal grades (cut, color, clarity) are encoded by their natural quality order, or one-hot encoded for the neural network.
- **Log-transformed target.** Models are trained on log-price to handle the right-skewed distribution, and predictions are transformed back so that all metrics are reported in USD.
- **Reproducible experiments.** Fixed random seeds, a stratified train/test split on carat quintiles, and pinned dependency versions.
- **Disciplined evaluation protocol.** The held-out test set is never touched during model development; it is evaluated only once, through an explicit `--final-eval` flag.
- **Model registry pattern.** All models are created through a single factory (`MODEL_BUILDERS`), so adding a model requires no change to the training, tuning or diagnostics scripts.
- **Model persistence.** The selected model is saved as a checkpoint together with a JSON file describing its selection metric and performance.
- **Automated tests.** Unit tests cover data validation and cleaning, preprocessing pipelines and model fit/predict behaviour.
- **Model analysis.** Learning and loss curves, residual diagnostics, predicted-versus-actual plots and feature importance support the interpretation of results.

## Project Structure

```text
.
├── data/raw/                          Raw dataset
├── data_analysis/
│   └── eda.py                         Exploratory data analysis
├── src/diamond_price_prediction/
│   ├── data.py                        Loading, validation, cleaning, encoding, splitting
│   ├── preprocessing.py               Feature preprocessing pipelines
│   ├── models.py                      Model factory and target transformation
│   └── evaluation.py                  Regression metrics and reporting helpers
├── scripts/
│   ├── train.py                       Cross-validated training and model selection
│   ├── tune.py                        Hyperparameter grid search
│   ├── loss_curves.py                 Learning and loss curves
│   ├── diagnostics.py                 Cross-validated diagnostics and feature importance
│   └── predict.py                     Inference check with the saved model
├── models/                            Saved model checkpoint and metadata
├── reports/
│   ├── preliminary/                   Preliminary documentation
│   ├── midterm/                       Midterm report
│   ├── final/                         Final report
│   ├── figures/                       Generated figures
│   └── tables/                        Generated metric tables
├── docs/                              Course and project instructions
├── tests/                             Automated tests
└── requirements.txt                   Pinned dependencies
```

## Methodology

1. **Data preparation**: validation, cleaning, ordinal encoding and a stratified 80/20 split (43,017 training rows, 10,755 held-out rows).
2. **Preprocessing**: log transform of `carat`, optional scaling, and ordinal or one-hot encoding depending on the model.
3. **Models**: mean baseline, linear regression, Random Forest, XGBoost and an MLP regressor, all trained on log-price.
4. **Model selection**: five-fold cross-validation on the training split, selecting by mean RMSE.
5. **Final evaluation**: a single evaluation of the fitted models on the held-out test set.
6. **Analysis**: learning curves, residuals, predicted versus actual values and feature importance.

## Setup

Requires Python 3.11 or higher.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root:

```bash
PYTHONPATH=src python data_analysis/eda.py        # exploratory analysis
PYTHONPATH=src python scripts/train.py            # cross-validation and checkpoint
PYTHONPATH=src python scripts/tune.py             # hyperparameter search
PYTHONPATH=src python scripts/loss_curves.py      # learning and loss curves
PYTHONPATH=src python scripts/diagnostics.py      # diagnostics and feature importance
PYTHONPATH=src python scripts/predict.py          # inference with the saved model
```

By default, `train.py` performs five-fold cross-validation on the training split and does not touch the held-out test set. The final evaluation is run explicitly with:

```bash
PYTHONPATH=src python scripts/train.py --final-eval
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Documentation

- [Final report](reports/final/final_report.pdf): complete methodology, experiments and analysis.
- [Midterm report](reports/midterm/midterm_report.pdf)
- [Preliminary documentation](reports/preliminary/preliminary-documentation.md): project scope, dataset and planned algorithms.

## Authors

Anatoli Abahumna and Leonardo Meloni, project for Warsaw University of Technology EARIN course
