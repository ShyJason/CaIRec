import types
import unittest

import torch

from model import MILK_model


class RecommenderBridgeTest(unittest.TestCase):
    def test_decoupled_bridge_keeps_observed_and_adapts_missing_features(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 2
        model.env = types.SimpleNamespace(device="cpu")
        model.disable_imputation = False

        raw = {
            "v": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            "t": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        }
        recommendation = {
            "v": torch.full((2, 2), 10.0),
            "t": torch.full((2, 2), 20.0),
        }
        completed = {
            "v": torch.full((2, 2), 3.0),
            "t": torch.full((2, 2), 4.0),
        }
        adapted = {
            "v": torch.full((2, 2), 30.0),
            "t": torch.full((2, 2), 40.0),
        }

        model.project_recommendation_features = lambda raw_features: recommendation
        model.project_features = lambda raw_features: completed
        model._build_completed_features = lambda *args, **kwargs: completed
        model.adapt_completed_to_recommendation = lambda value: adapted

        result = model.get_recommender_modal_features(raw_features=raw)
        self.assertTrue(torch.equal(result["v"][0], recommendation["v"][0]))
        self.assertTrue(torch.equal(result["v"][1], adapted["v"][1]))
        self.assertTrue(torch.equal(result["t"][0], adapted["t"][0]))
        self.assertTrue(torch.equal(result["t"][1], recommendation["t"][1]))


if __name__ == "__main__":
    unittest.main()
