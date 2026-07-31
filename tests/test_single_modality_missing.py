import io
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from dataset_loader import Loader4MM


class SingleModalityMissingTest(unittest.TestCase):
    DATASET_SIZES = {
        "clothing": 23033,
        "beauty": 11161,
        "sports": 18357,
    }

    def _load(self, dataset):
        loader = Loader4MM.__new__(Loader4MM)
        with tempfile.TemporaryDirectory() as data_path:
            loader.env = SimpleNamespace(
                DATA_PATH=data_path,
                args=SimpleNamespace(dataset=dataset),
            )
            n_items = self.DATASET_SIZES[dataset]
            loader.trainItem = list(range(n_items))
            features = [
                np.ones((n_items, 1), dtype=np.float32),
                np.ones((n_items, 1), dtype=np.float32),
            ]
            with redirect_stdout(io.StringIO()):
                loader.load_missing_payload(features, rate=0.5)
        return loader

    def test_bundled_payload_is_shared_across_all_phases(self):
        for dataset in self.DATASET_SIZES:
            with self.subTest(dataset=dataset):
                loader = self._load(dataset)
                expected = loader.train_missing_modality_items
                self.assertEqual(
                    len(expected["items"]),
                    self.DATASET_SIZES[dataset] // 2,
                )
                self.assertTrue(np.all(np.isin(expected["indicator"], (0, 1))))
                for name in (
                    "eval_val_missing_modality_items",
                    "test_missing_modality_items",
                ):
                    actual = getattr(loader, name)
                    np.testing.assert_array_equal(actual["items"], expected["items"])
                    np.testing.assert_array_equal(
                        actual["indicator"], expected["indicator"]
                    )
                self.assertEqual(len(loader.val_missing_modality_items["items"]), 0)

    def test_payload_rejects_incompatible_item_count(self):
        loader = Loader4MM.__new__(Loader4MM)
        with tempfile.TemporaryDirectory() as data_path:
            loader.env = SimpleNamespace(
                DATA_PATH=data_path,
                args=SimpleNamespace(dataset="clothing"),
            )
            loader.trainItem = list(range(10))
            features = [
                np.ones((10, 1), dtype=np.float32),
                np.ones((10, 1), dtype=np.float32),
            ]
            with self.assertRaisesRegex(ValueError, "invalid or duplicate item ids"):
                loader.load_missing_payload(features, rate=0.5)


if __name__ == "__main__":
    unittest.main()
