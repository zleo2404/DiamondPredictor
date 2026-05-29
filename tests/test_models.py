import unittest

from diamond_price_prediction.data import load_and_prepare
from diamond_price_prediction.models import build_linear_regression, build_mean_baseline


class ModelSmokeTests(unittest.TestCase):
    def test_baseline_fit_predicts_requested_number_of_rows(self):
        X_train, _, y_train, _ = load_and_prepare()
        model = build_mean_baseline()

        model.fit(X_train.head(50), y_train.head(50))
        predictions = model.predict(X_train.head(5))

        self.assertEqual(len(predictions), 5)

    def test_linear_regression_pipeline_fit_predicts_positive_prices(self):
        X_train, _, y_train, _ = load_and_prepare()
        model = build_linear_regression()

        model.fit(X_train.head(200), y_train.head(200))
        predictions = model.predict(X_train.head(5))

        self.assertEqual(len(predictions), 5)
        self.assertTrue((predictions > 0).all())


if __name__ == "__main__":
    unittest.main()
