import unittest
from types import SimpleNamespace

import torch

from model import MILK_model


class RecNeighborCLTest(unittest.TestCase):
    def test_mainline_uses_cross_modal_positive_and_same_modal_negatives(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 2
        model.disable_imputation = False
        model.env = SimpleNamespace(device=torch.device("cpu"))
        raw = {"v": torch.zeros(2, 2), "t": torch.ones(2, 2)}
        model._current_raw_modal_features = lambda: raw
        model._missing_masks = lambda raw_features: {
            "v": torch.tensor([False, True]),
            "t": torch.tensor([True, True]),
        }
        model._recommendation_gcn_modality_item_embeddings = lambda **kwargs: {
            "v": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "t": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }

        loss = model.compute_true_missing_gcn_infonce_loss(
            torch.tensor([0, 1]),
            temperature=0.2,
            bank_size=2,
        )

        self.assertGreater(float(loss), 0.6)


if __name__ == "__main__":
    unittest.main()
