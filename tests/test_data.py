import unittest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diamond_price_prediction.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean,
    encode_ordinals,
    load_raw,
    split,
    validate_raw,
)


class DataPipelineTests(unittest.TestCase):
    def test_raw_dataset_has_required_columns_and_no_missing_values(self):
        df = load_raw()

        self.assertTrue(set(NUMERIC_FEATURES).issubset(df.columns))
        self.assertTrue(set(CATEGORICAL_FEATURES).issubset(df.columns))
        self.assertIn(TARGET, df.columns)
        required_data = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]
        self.assertFalse(required_data.isna().any().any())

    def test_validate_raw_rejects_missing_required_column(self):
        df = load_raw().drop(columns=[TARGET])

        with self.assertRaises(ValueError):
            validate_raw(df)

    def test_clean_removes_invalid_dimensions(self):
        df = pd.DataFrame(
            {
                "carat": [1.0, 1.0, 1.0, 1.0],
                "cut": ["Ideal"] * 4,
                "color": ["E"] * 4,
                "clarity": ["VS1"] * 4,
                "depth": [61.0] * 4,
                "table": [57.0] * 4,
                "price": [5000] * 4,
                "x": [6.0, 0.0, 6.0, 6.0],
                "y": [6.0, 6.0, 31.0, 6.0],
                "z": [4.0, 4.0, 4.0, 31.0],
            }
        )

        cleaned = clean(df)

        self.assertEqual(len(cleaned), 1)
        self.assertTrue((cleaned[["x", "y", "z"]] > 0).all(axis=None))

    def test_encoding_produces_numeric_categories(self):
        encoded = encode_ordinals(load_raw().head(20))

        for column in CATEGORICAL_FEATURES:
            self.assertTrue(pd.api.types.is_numeric_dtype(encoded[column]))

    def test_split_keeps_train_and_test_indices_separate(self):
        df = encode_ordinals(clean(load_raw()))
        X_train, X_test, y_train, y_test = split(df)

        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertTrue(set(X_train.index).isdisjoint(set(X_test.index)))


if __name__ == "__main__":
    unittest.main()
