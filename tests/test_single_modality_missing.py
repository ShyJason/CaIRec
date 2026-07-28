import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dataset_loader import Loader4MM


def build_loader(policy="random", missing_rate=0.3, eval_missing_rate=0.5):
    loader = Loader4MM.__new__(Loader4MM)
    loader.env = SimpleNamespace(
        args=SimpleNamespace(
            dataset_seed=0,
            missing_mask_protocol="i3",
            train_missing_modality=policy,
            missing_rate=missing_rate,
            eval_missing_rate=eval_missing_rate,
            imputation_val_rate=0.1,
        )
    )
    loader.trainItem = list(range(100))
    loader.valItem = np.arange(100, 140)
    loader.testItem = np.arange(140, 180)
    features = [np.ones((200, 4), dtype=np.float32) for _ in range(2)]
    with redirect_stdout(io.StringIO()):
        loader.set_miss_mutimedia_feature_items(features, seed=0, rate=missing_rate)
    return loader


class SingleModalityMissingTest(unittest.TestCase):
    def test_explicit_unified_payload_keeps_one_modality_across_all_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload_path = Path(tmp) / "fixed_text.npy"
            items = np.array([2, 5, 8, 13], dtype=np.int64)
            np.save(
                payload_path,
                {
                    "protocol": "unified_single_modality",
                    "dataset": "clothing",
                    "missing_rate": 0.2,
                    "items": items,
                    "indicator": np.ones(items.shape, dtype=np.int64),
                },
                allow_pickle=True,
            )
            loader = Loader4MM.__new__(Loader4MM)
            loader.env = SimpleNamespace(
                DATA_PATH=tmp,
                args=SimpleNamespace(
                    dataset="clothing",
                    seed=2023,
                    unified_payload_seed=2023,
                    unified_payload_file=payload_path.name,
                    dataset_seed=0,
                    missing_mask_protocol="unified_static",
                    train_missing_modality="text",
                    eval_missing_rate=0.2,
                ),
            )
            loader.trainItem = list(range(20))
            features = [np.ones((20, 4), dtype=np.float32) for _ in range(2)]
            with redirect_stdout(io.StringIO()):
                loader.set_miss_mutimedia_feature_items(features, rate=0.2)
            for name in (
                "train_missing_modality_items",
                "eval_val_missing_modality_items",
                "test_missing_modality_items",
            ):
                metadata = getattr(loader, name)
                np.testing.assert_array_equal(metadata["items"], items)
                np.testing.assert_array_equal(metadata["indicator"], np.ones_like(items))

    def test_text_only_applies_to_training_and_stage1_holdout(self):
        loader = build_loader("text", missing_rate=0.5)
        self.assertTrue(np.all(loader.train_missing_modality_items["indicator"] == 1))
        self.assertTrue(np.all(loader.val_missing_modality_items["indicator"] == 1))

        dynamic = loader.sample_stage1_dynamic_missing_metadata(seed=91, rate=0.7)
        self.assertTrue(np.all(dynamic["indicator"] == 1))

    def test_image_only_applies_to_training_and_stage1_holdout(self):
        loader = build_loader("image", missing_rate=0.5)
        self.assertTrue(np.all(loader.train_missing_modality_items["indicator"] == 0))
        self.assertTrue(np.all(loader.val_missing_modality_items["indicator"] == 0))

        dynamic = loader.sample_stage1_dynamic_missing_metadata(seed=92, rate=0.7)
        self.assertTrue(np.all(dynamic["indicator"] == 0))

    def test_recommendation_validation_and_test_stay_random(self):
        loader = build_loader("text", missing_rate=0.3, eval_missing_rate=0.5)
        for metadata in (
            loader.eval_val_missing_modality_items,
            loader.test_missing_modality_items,
        ):
            selected = metadata["items"]
            np.testing.assert_array_equal(
                metadata["indicator"],
                loader.protected_indices[selected],
            )
            self.assertEqual(len(selected), 20)

    def test_random_policy_is_backward_compatible(self):
        baseline = build_loader("random", missing_rate=0.3)
        repeated = build_loader("random", missing_rate=0.3)
        for name in (
            "train_missing_modality_items",
            "val_missing_modality_items",
            "eval_val_missing_modality_items",
            "test_missing_modality_items",
        ):
            lhs = getattr(baseline, name)
            rhs = getattr(repeated, name)
            np.testing.assert_array_equal(lhs["items"], rhs["items"])
            np.testing.assert_array_equal(lhs["indicator"], rhs["indicator"])

    def test_rate_counts_and_validation(self):
        for rate in (0.1, 0.3, 0.5, 0.7, 0.9):
            loader = build_loader("image", missing_rate=rate)
            # Ten percent of training items are reserved for Stage 1 validation.
            self.assertEqual(
                len(loader.train_missing_modality_items["items"]),
                int(90 * rate),
            )

        with self.assertRaisesRegex(ValueError, "missing_rate"):
            build_loader("image", missing_rate=1.1)
        with self.assertRaisesRegex(ValueError, "eval_missing_rate"):
            build_loader("image", eval_missing_rate=-0.1)
        with self.assertRaisesRegex(ValueError, "train_missing_modality"):
            build_loader("audio")


if __name__ == "__main__":
    unittest.main()
