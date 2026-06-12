# Diamond Price Prediction

This project predicts diamond prices from physical measurements and quality
grades using the standard diamonds dataset. The task is supervised regression:
the target is `price`, and the input features are `carat`, `cut`, `color`,
`clarity`, `depth`, `table`, `x`, `y`, and `z`.

## Project Structure

```text
data/raw/diamonds.csv              Raw dataset
src/diamond_price_prediction/      Reusable data, preprocessing, model, and metric code
data_analysis/eda.py               Exploratory analysis script
scripts/train.py                   Cross-validation training and checkpoint generation
scripts/loss_curves.py             Learning/loss curve generation
scripts/diagnostics.py             Cross-validated diagnostics and feature importance
scripts/predict.py                 Saved-model inference check
reports/figures/                   Generated figures
reports/tables/                    Generated metric tables
models/                            Saved model checkpoints
tests/                             Automated checks
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Reproduce Outputs

Run commands from the repository root:

```bash
PYTHONPATH=src python data_analysis/eda.py
PYTHONPATH=src python scripts/train.py
PYTHONPATH=src python scripts/loss_curves.py
PYTHONPATH=src python scripts/diagnostics.py
PYTHONPATH=src python scripts/predict.py
```

The default training command performs 5-fold cross-validation on the training
split and does not evaluate the held-out test split. The final test evaluation
is done:

```bash
PYTHONPATH=src python scripts/train.py --final-eval
```


## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
