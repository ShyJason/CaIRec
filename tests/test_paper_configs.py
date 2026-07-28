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
                    self.assertEqual(config["missing_mask_protocol"], "unified_static")
                    self.assertEqual(config["missing_rate"], 0.5)
                    self.assertEqual(config["eval_missing_rate"], 0.5)
                    self.assertEqual(config["unified_payload_seed"], 2023)
                    self.assertEqual(config["feature_bridge_mode"], "decoupled_latent")

    def test_stage2_matches_recorded_dataset_specific_settings(self):
        expected = {
            "clothing": (0, 50, 0.005, "posterior_reliability", "fusion"),
            "beauty": (2023, 20, 0.010, "mean", None),
            "sports": (0, 30, 0.005, "posterior_reliability", "both"),
        }
        for dataset, values in expected.items():
            with self.subTest(dataset=dataset):
                config = self._load(dataset, "stage2")
                dataset_seed, patience, cl_weight, fusion, scope = values
                self.assertEqual(config["dataset_seed"], dataset_seed)
                self.assertEqual(config["early_stop"], patience)
                self.assertEqual(config["rec_neighbor_cl_weight"], cl_weight)
                self.assertEqual(config["fusion_mode"], fusion)
                if scope is not None:
                    self.assertEqual(config["posterior_reliability_scope"], scope)
                self.assertEqual(config["item_graph_kind"], "modality_completed")
                self.assertEqual(config["item_graph_topk"], 10)
                self.assertEqual(config["item_graph_cf_weight"], 0.4)
                self.assertEqual(config["item_graph_image_weight"], 0.6)
                self.assertEqual(config["item_graph_text_weight"], 0.6)
                self.assertEqual(config["item_graph_modal_alpha"], 0.25)


if __name__ == "__main__":
    unittest.main()
