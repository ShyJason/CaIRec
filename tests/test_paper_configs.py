import hashlib
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PaperConfigTest(unittest.TestCase):
    def _load(self, dataset, stage):
        path = ROOT / "configs" / dataset / f"paper_{stage}.yaml"
        self.assertTrue(path.is_file(), path)
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_all_datasets_have_the_three_stage_pipeline(self):
        for dataset in ("clothing", "beauty", "sports"):
            with self.subTest(dataset=dataset):
                stage1_1 = self._load(dataset, "stage1_1")
                stage1_2 = self._load(dataset, "stage1_2")
                stage2 = self._load(dataset, "stage2")
                self.assertEqual(stage1_1["train_stage"], "imputer_param")
                self.assertEqual(stage1_2["train_stage"], "imputer_backprop")
                self.assertEqual(stage2["train_stage"], "recommender")
                for config in (stage1_1, stage1_2, stage2):
                    self.assertEqual(config["missing_rate"], 0.5)
                    self.assertEqual(config["feature_bridge_mode"], "decoupled_latent")

    def test_bundled_missing_payloads_match_the_paper_protocol(self):
        expected = {
            "clothing": "34e09412a337e19906b16bb7bdb9e097d824e1e85a1b1908e501e5a29bc1873c",
            "beauty": "408e3c8bfffd8322412e63b77cc87b07ae4ce1f02329f0500d61c2aee95e0cf4",
            "sports": "421816fbeaa65cb6323f9f42e209a52f5688401525ba75bb6c902789580aaabe",
        }
        for dataset, expected_hash in expected.items():
            with self.subTest(dataset=dataset):
                path = (
                    ROOT
                    / "configs"
                    / dataset
                    / "unified_missing_items_mr0.5_seed2023.npy"
                )
                self.assertTrue(path.is_file(), path)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)

    def test_stage2_matches_canonical_dataset_specific_settings(self):
        expected = {
            "clothing": 50,
            "beauty": 20,
            "sports": 30,
        }
        for dataset, patience in expected.items():
            with self.subTest(dataset=dataset):
                config = self._load(dataset, "stage2")
                self.assertEqual(config["early_stop"], patience)
                self.assertNotIn("rec_neighbor_cl_weight", config)
                self.assertNotIn("rec_neighbor_cl_temp", config)
                self.assertNotIn("rec_neighbor_cl_bank_size", config)
                self.assertEqual(config["item_graph_kind"], "modality_completed")
                self.assertEqual(config["item_graph_topk"], 10)
                self.assertEqual(config["item_graph_cf_weight"], 0.4)
                self.assertEqual(config["item_graph_image_weight"], 0.6)
                self.assertEqual(config["item_graph_text_weight"], 0.6)
                self.assertEqual(config["item_graph_modal_alpha"], 0.25)


if __name__ == "__main__":
    unittest.main()
