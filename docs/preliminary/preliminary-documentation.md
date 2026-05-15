# Preliminary Project Documentation: Diamond Price Prediction

**Authors:** Anatoli Abahumna and Leonardo Meloni  
**Group:** 16  
**Warsaw University of Technology**

## Project Scope

The goal is to predict the sale price of a diamond from its measurable properties and quality grades. This is a supervised regression problem: each training example contains input features such as carat, cut, color, clarity, depth, table, and dimensions, and the target value is the diamond price in US dollars.

The solution will implement Linear/Ridge/Bayesian Ridge Regression, Random Forest, and XGBoost/Gradient Boosting. These models are appropriate for tabular data with a moderate number of rows and features, and they keep the method explainable.

## Selected Dataset

The primary dataset will be the standard `diamonds` dataset documented by ggplot2. It contains 53,940 diamonds and 10 variables: `price`, `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, and `z` [1][2].

The target variable is:

- `price`: diamond price in US dollars.

The input features are:

• Numerical: carat (weight), depth (total depth percentage), table (width of top relative to widest point), and physical dimensions x (length), y (width), z (depth).

• Categorical: cut (quality of the cut), color (diamond color, from J to D), and clarity (measurement of how clear the diamond is).


The dataset is suitable because it is large enough for meaningful train/test evaluation, small enough for quick experiments on a laptop, and contains both physical measurements and quality categories. During preparation, the data will be checked for missing values, duplicate rows, zero dimensions, and unusual outliers. If a Kaggle version is used, index-only columns such as `id` or `Unnamed: 0` will be removed before training.

## Planned Algorithms

### Baseline: Mean Price Regressor

A simple baseline will always predict the average training price. It provides a minimum reference point; all trained models should improve over it in MAE, RMSE, and R2.

Example: if the training-set average price is 3900 USD, the baseline predicts 3900 USD for every diamond, regardless of carat or clarity.

### Linear and Ridge Regression

Linear Regression estimates one weight per feature and gives a direct baseline for how much each encoded feature contributes to price. Ridge Regression extends it with L2 regularization, which penalizes very large coefficients [3]. Both will be used after scaling numeric features and encoding categorical features.

Example: after preprocessing, Linear Regression may learn that higher `carat` strongly increases predicted price. Ridge Regression should give a similar interpretation while keeping correlated size features such as `x`, `y`, and `z` from receiving unstable coefficients.

### Bayesian Ridge Regression

Bayesian Ridge Regression provides a probabilistic approach to linear modeling. Instead of estimating a single, fixed point value for each feature weight, it assumes a prior probability distribution over the weights and updates them based on the data to form a posterior distribution. Furthermore, it automatically estimates its own regularization parameters from the data during training, reducing the need for extensive hyperparameter tuning [9].

Example: unlike standard Linear Regression which predicts a single fixed number, Bayesian Ridge estimates a probability distribution for the price. For a high-carat diamond with rare clarity, it might predict a mean price of 15,000 USD along with a standard deviation of 400 USD. This provides a measurable confidence interval, allowing us to quantify how uncertain the model is about that specific prediction.

### Random Forest Regressor

Random Forest trains many decision trees on different samples of the data and averages their predictions. This usually improves accuracy and reduces overfitting compared with a single decision tree [4].

Example: one tree may first split diamonds by `carat`, another by `clarity`, and another by `x` or `z`. The final prediction is the average of all tree predictions.

### XGBoost / Gradient Boosting Regressor

Gradient Boosting trains trees sequentially, where each new tree corrects errors made by the previous trees. XGBoost is an efficient implementation of this idea and supports regression tasks [5].

Example: early trees may capture the broad price increase caused by `carat`, while later trees correct systematic errors for high-clarity or unusually proportioned diamonds.

The expected main candidates are Random Forest and XGBoost/Gradient Boosting. Linear, Ridge and Bayesian Ridge Regression will be retained because they are easy to explain and provide useful sanity checks.

## Libraries and Tools

The implementation will use:

- Python: required by the project instructions.
- pandas and NumPy: data loading, cleaning, and numeric operations.
- scikit-learn: preprocessing, train/test splitting, pipelines, regression models, cross-validation, and metrics.
- XGBoost: gradient boosting regression model.
- matplotlib and seaborn: exploratory plots and result visualizations.
- joblib: saving the final trained model.
- pytest: basic automated tests for data loading, preprocessing, and model prediction.
- Faculty GitLab: version control and final code submission, as required by the project-specific instructions.

The code will use scikit-learn `Pipeline` and `ColumnTransformer` so preprocessing and model training are kept together and cross-validated correctly [6][7]. This also reduces the risk of training/test data leakage.

## Experiment and Test Plan

The experiments will follow a fixed, reproducible workflow:

1. Load and preprocess the dataset.
2. Perform exploratory data analysis.
3. Split the data into training and final test sets, using a fixed random seed.
4. Fit the mean-price baseline.
5. Train Linear/Ridge/Bayesian Ridge Regression, Random Forest, and XGBoost/Gradient Boosting models.
6. Tune important hyperparameters using cross-validation on the training set.
7. Evaluate the models on the held-out test set.
8. Save the final model and record the exact metrics and plots and compare them.

Planned experiments:

- Compare raw target prediction with `log1p(price)` target transformation. The log target may reduce the effect of very expensive outliers; final metrics will still be reported in dollars after inverse transformation.
- Compare categorical encoding strategies: explicit ordered encoding for `cut`, `color`, and `clarity`, and one-hot encoding for the linear baseline if needed.
- Compare feature sets: numeric-only features versus numeric plus categorical features.
- Compare model complexity: shallow versus deeper trees, and small versus moderate ensemble sizes.

Planned tests:

- Dataset tests: required columns exist, target is positive, no unexpected missing values remain after preprocessing.
- Cleaning tests: invalid zero dimensions are handled consistently.
- Split tests: training and test rows do not overlap.
- Pipeline tests: each model can fit and predict on a small sample.
- Persistence tests: the saved model can be loaded and used for inference.

## Result Visualization

The report will include:

- Histogram of `price` and `log(price)` to show target skew.
- Scatter plot of `carat` versus `price`, optionally sampled for readability.
- Box plots of `price` by `cut`, `color`, and `clarity`.
- Correlation heatmap for numeric features.
- Bar chart comparing model MAE, RMSE, and R2.
- Actual-versus-predicted scatter plot for the final model.
- Residual plots against predicted price and `carat`.
- Feature importance or permutation importance for the best tree-based model.

These plots will support model comparison and error analysis.

## Evaluation Metrics

The main quality measure will be Mean Absolute Error because it is directly interpretable in dollars.

- MAE: average absolute prediction error. Represents the average absolute difference between the predicted and actual diamond prices. It is highly interpretable (e.g., "The model is off by X dollars on average").
- RMSE: Measures the standard deviation of the residuals. It penalizes larger errors more heavily, which is crucial to ensure the model does not fail spectacularly on high-value diamonds.
- R2: It explains the proportion of the variance in the diamond price that is predictable from the given features (a score closer to 1.0 indicates a better fit).
- MAPE: average percentage error. It will be reported only as a secondary metric because percentage errors can overemphasize cheaper diamonds.

Model selection will be based primarily on cross-validated MAE. Final model quality will be reported on the held-out test set using all metrics. The baseline mean-price regressor will be included in every result table so improvements are measurable.

## References

[1] ggplot2 documentation, "Prices of over 50,000 round cut diamonds": https://ggplot2.tidyverse.org/reference/diamonds.html 
[2] kaggle dataset link: https://www.kaggle.com/datasets/shivam2503/diamonds
[3] scikit-learn documentation, `Ridge`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html  
[4] scikit-learn documentation, `RandomForestRegressor`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html  
[5] XGBoost documentation: https://xgboost.readthedocs.io/en/stable/index.html  
[6] scikit-learn documentation, `Pipeline`: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html  
[7] scikit-learn documentation, `ColumnTransformer`: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html  
[8] scikit-learn documentation, regression metrics and cross-validation: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics and https://scikit-learn.org/stable/modules/cross_validation.html
[9] scikit-learn documentation, Bayesian Ridge Regressor: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
