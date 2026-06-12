import unittest

import numpy as np

from diamond_price_prediction.data import CATEGORICAL_FEATURES, load_and_prepare
from diamond_price_prediction.preprocessing import build_feature_preprocessor


class PreprocessingTests(unittest.TestCase):
    def test_default_scaled_preprocessor_keeps_ordinal_feature_count(self):
        X_train, _, _, _ = load_and_prepare()
        transformed = build_feature_preprocessor(scale_numeric=True).fit_transform(
            X_train.head(25)
        )

        self.assertEqual(transformed.shape[1], 9)

    def test_onehot_preprocessor_expands_categorical_feature_count(self):
        X_train, _, _, _ = load_and_prepare()
        transformed = build_feature_preprocessor(
            scale_numeric=True,
            categorical="onehot",
        ).fit_transform(X_train.head(25))

        self.assertEqual(transformed.shape[1], 26)

    def test_invalid_categorical_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_feature_preprocessor(scale_numeric=True, categorical="bad-mode")

    def test_ordinal_categorical_values_pass_through_unchanged(self):
        X_train, _, _, _ = load_and_prepare()
        sample = X_train.head(25)
        transformed = build_feature_preprocessor(
            scale_numeric=False,
            categorical="ordinal",
        ).fit_transform(sample)

        np.testing.assert_array_equal(
            transformed[:, -len(CATEGORICAL_FEATURES) :],
            sample[CATEGORICAL_FEATURES].to_numpy(),
        )


if __name__ == "__main__":
    unittest.main()
