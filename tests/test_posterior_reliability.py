import math
import unittest

import torch

from model import MILK_model


class PosteriorReliabilityTest(unittest.TestCase):
    def _model(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.d_beta = 1
        model.promrl_dim = 2
        model.W = torch.nn.ParameterDict(
            {
                "v": torch.nn.Parameter(torch.tensor([[1.0], [0.0]]), requires_grad=False),
                "t": torch.nn.Parameter(torch.tensor([[1.0], [0.0]]), requires_grad=False),
            }
        )
        model.log_sigma = torch.nn.ParameterDict(
            {
                "v": torch.nn.Parameter(torch.zeros(1), requires_grad=False),
                "t": torch.nn.Parameter(torch.zeros(1), requires_grad=False),
            }
        )
        model.posterior_reliability_scale = 1.0
        model.posterior_reliability_floor = 0.0
        model.use_posterior_reliability = True
        model.posterior_reliability_scope = "both"
        return model

    def test_scope_can_separate_graph_and_fusion(self):
        model = self._model()
        model.posterior_reliability_scope = "graph"
        self.assertTrue(model._uses_posterior_reliability_for("graph"))
        self.assertFalse(model._uses_posterior_reliability_for("fusion"))
        model.posterior_reliability_scope = "fusion"
        self.assertFalse(model._uses_posterior_reliability_for("graph"))
        self.assertTrue(model._uses_posterior_reliability_for("fusion"))

    def test_predictive_variance_reliability_by_missing_pattern(self):
        model = self._model()
        masks = {
            "v": torch.tensor([True, False, True]),
            "t": torch.tensor([False, True, True]),
        }

        reliability = model._posterior_completion_reliabilities(masks)

        # V=(I + W^T W)^-1=0.5 and mean predictive variance is
        # tr(W V W^T)/2 + sigma^2 = 0.25 + 1 = 1.25.
        expected_missing = math.exp(-1.25)
        self.assertAlmostEqual(float(reliability["t"][0]), expected_missing, places=6)
        self.assertAlmostEqual(float(reliability["v"][1]), expected_missing, places=6)
        self.assertEqual(float(reliability["v"][0]), 1.0)
        self.assertEqual(float(reliability["t"][1]), 1.0)
        self.assertEqual(float(reliability["v"][2]), 1.0)
        self.assertEqual(float(reliability["t"][2]), 1.0)

    def test_fusion_normalizes_reliability_across_modalities(self):
        model = self._model()
        masks = {"v": torch.tensor([True]), "t": torch.tensor([False])}
        item_outputs = {"v": torch.tensor([[2.0]]), "t": torch.tensor([[10.0]])}

        fused = model._fuse_item_sources(item_outputs, observed_masks=masks)

        missing = math.exp(-1.25)
        expected = (2.0 + missing * 10.0) / (1.0 + missing)
        self.assertAlmostEqual(float(fused[0, 0]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
