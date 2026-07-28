import unittest
from types import SimpleNamespace

import torch

from model import MILK_model


class RecNeighborCLOptionsTest(unittest.TestCase):
    def test_frontend_stage_skips_all_graph_propagation(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v"]
        raw = {"v": torch.tensor([[3.0, 4.0], [0.0, 2.0]])}
        model._current_raw_modal_features = lambda: raw
        model.get_recommender_modal_features = lambda raw_features, allow_imputer_grad: raw_features
        model._gcn_skip_mlp = lambda: False

        class FrontendOnlyGCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.MLP = torch.nn.Identity()

            def forward(self, *args, **kwargs):
                raise AssertionError("frontend CL must not call graph propagation")

        model.v_gcn = FrontendOnlyGCN()
        outputs = model._recommendation_frontend_modality_item_embeddings()
        expected = torch.nn.functional.normalize(raw["v"], dim=-1)
        self.assertTrue(torch.allclose(outputs["v"], expected))

    def test_post_item_graph_stage_applies_modality_residual(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 2
        model.user_emb = SimpleNamespace(weight=torch.zeros(1, 2))
        model._current_raw_modal_features = lambda: {
            "v": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "t": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        model.get_recommender_modal_features = lambda raw_features, allow_imputer_grad: raw_features
        model._missing_masks = lambda raw_features: {
            "v": torch.ones(2, dtype=torch.bool),
            "t": torch.ones(2, dtype=torch.bool),
        }
        model._gcn_skip_mlp = lambda: True

        class IdentityGCN(torch.nn.Module):
            def forward(self, features, user_id_emb, skip_mlp=False):
                return user_id_emb, features

        model.v_gcn = IdentityGCN()
        model.t_gcn = IdentityGCN()
        calls = []

        def residual(item_emb, observed_mask=None, modality=None):
            calls.append(modality)
            return item_emb.flip(0)

        model._apply_item_graph_modal_residual = residual
        outputs = model._recommendation_gcn_modality_item_embeddings(apply_item_graph=True)
        self.assertEqual(calls, ["v", "t"])
        self.assertTrue(torch.equal(outputs["v"], torch.tensor([[0.0, 1.0], [1.0, 0.0]])))

    def test_cf_neighbor_uses_observed_same_modality_teacher(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 2
        model.disable_imputation = False
        model.env = SimpleNamespace(
            device=torch.device("cpu"),
            args=SimpleNamespace(
                rec_neighbor_cl_stage="gcn",
                rec_neighbor_cl_anchor_weighting="uniform",
                rec_neighbor_cl_false_negative_threshold=1.1,
                rec_neighbor_cl_objective="positive_cosine",
                rec_neighbor_cl_positive_source="cf_neighbor",
            ),
        )
        raw = {
            "v": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            "t": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        }
        masks = {
            "v": torch.tensor([False, True]),
            "t": torch.tensor([True, True]),
        }
        model._current_raw_modal_features = lambda: raw
        model._missing_masks = lambda raw_features: masks
        model._recommendation_gcn_modality_item_embeddings = lambda **kwargs: {
            "v": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "t": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        }
        model.ItemItemGraphs = {
            "cf": torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [1, 0]]),
                torch.ones(2),
                (2, 2),
            ).coalesce(),
        }

        loss = model.compute_true_missing_gcn_infonce_loss(
            torch.tensor([0, 1]),
            temperature=0.2,
            bank_size=2,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_positive_cosine_does_not_sample_unused_negative_bank(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 3
        model.disable_imputation = False
        model.env = SimpleNamespace(
            device=torch.device("cpu"),
            args=SimpleNamespace(
                rec_neighbor_cl_stage="gcn",
                rec_neighbor_cl_anchor_weighting="uniform",
                rec_neighbor_cl_false_negative_threshold=1.1,
                rec_neighbor_cl_objective="positive_cosine",
                rec_neighbor_cl_positive_source="cross_modal",
            ),
        )
        raw = {"v": torch.zeros(3, 2), "t": torch.ones(3, 2)}
        masks = {
            "v": torch.tensor([False, False, False]),
            "t": torch.tensor([True, True, True]),
        }
        model._current_raw_modal_features = lambda: raw
        model._missing_masks = lambda raw_features: masks
        model._recommendation_gcn_modality_item_embeddings = lambda **kwargs: {
            "v": torch.nn.functional.normalize(torch.ones(3, 2), dim=-1),
            "t": torch.nn.functional.normalize(torch.ones(3, 2), dim=-1),
        }

        before = torch.random.get_rng_state()
        loss = model.compute_true_missing_gcn_infonce_loss(
            torch.tensor([0, 1, 2]), temperature=0.2, bank_size=1
        )
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_cross_modal_negative_bank_uses_positive_key_space(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 2
        model.disable_imputation = False
        args = SimpleNamespace(
            rec_neighbor_cl_stage="gcn",
            rec_neighbor_cl_anchor_weighting="uniform",
            rec_neighbor_cl_false_negative_threshold=1.1,
            rec_neighbor_cl_objective="infonce",
            rec_neighbor_cl_positive_source="cross_modal",
            rec_neighbor_cl_negative_source="same_modal",
        )
        model.env = SimpleNamespace(device=torch.device("cpu"), args=args)
        raw = {"v": torch.zeros(2, 2), "t": torch.ones(2, 2)}
        masks = {
            "v": torch.tensor([False, True]),
            "t": torch.tensor([True, True]),
        }
        model._current_raw_modal_features = lambda: raw
        model._missing_masks = lambda raw_features: masks
        model._recommendation_gcn_modality_item_embeddings = lambda **kwargs: {
            # The same-modal negative is identical to the anchor, while the true
            # cross-modal negative is orthogonal to the positive text key.
            "v": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "t": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }

        same_modal_loss = model.compute_true_missing_gcn_infonce_loss(
            torch.tensor([0, 1]), temperature=0.2, bank_size=2
        )
        args.rec_neighbor_cl_negative_source = "cross_modal"
        cross_modal_loss = model.compute_true_missing_gcn_infonce_loss(
            torch.tensor([0, 1]), temperature=0.2, bank_size=2
        )

        self.assertGreater(float(same_modal_loss), 0.6)
        self.assertLess(float(cross_modal_loss), 0.01)
        self.assertLess(float(cross_modal_loss), float(same_modal_loss))

    def test_cl_directly_updates_anchor_frontend_and_detaches_target(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 2
        model.disable_imputation = False
        model.env = SimpleNamespace(
            device=torch.device("cpu"),
            args=SimpleNamespace(
                rec_neighbor_cl_stage="frontend",
                rec_neighbor_cl_anchor_weighting="uniform",
                rec_neighbor_cl_false_negative_threshold=1.1,
                rec_neighbor_cl_objective="infonce",
                rec_neighbor_cl_positive_source="cross_modal",
                rec_neighbor_cl_negative_source="cross_modal",
            ),
        )
        masks = {
            "v": torch.tensor([False, True]),
            "t": torch.tensor([True, True]),
        }
        frontend_v = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        frontend_t = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        model._current_raw_modal_features = lambda: {"v": frontend_v, "t": frontend_t}
        model._missing_masks = lambda raw_features: masks
        model._recommendation_frontend_modality_item_embeddings = lambda **kwargs: {
            "v": frontend_v,
            "t": frontend_t,
        }
        loss = model.compute_true_missing_gcn_infonce_loss(
            torch.tensor([0, 1]), temperature=0.2, bank_size=2
        )
        loss.backward()

        self.assertIsNotNone(frontend_v.grad)
        self.assertGreater(float(frontend_v.grad.abs().sum()), 0.0)
        self.assertIsNone(frontend_t.grad)

    def test_user_preference_cl_detaches_users_and_observed_target(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.m_item = 2
        model.n_user = 3
        model.disable_imputation = False
        model.env = SimpleNamespace(
            device=torch.device("cpu"),
            args=SimpleNamespace(
                rec_neighbor_cl_stage="frontend",
                rec_neighbor_cl_anchor_weighting="uniform",
                rec_neighbor_cl_false_negative_threshold=1.1,
                rec_neighbor_cl_objective="infonce",
                rec_neighbor_cl_positive_source="cross_modal",
                rec_neighbor_cl_negative_source="cross_modal",
                rec_neighbor_cl_similarity_space="user_preference",
            ),
        )
        model.user_emb = torch.nn.Embedding(3, 2)
        with torch.no_grad():
            model.user_emb.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )
        masks = {
            "v": torch.tensor([False, True]),
            "t": torch.tensor([True, True]),
        }
        frontend_v = torch.tensor(
            [[0.7, 0.7], [0.0, 1.0]],
            requires_grad=True,
        )
        frontend_t = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        model._current_raw_modal_features = lambda: {
            "v": frontend_v,
            "t": frontend_t,
        }
        model._missing_masks = lambda raw_features: masks
        model._recommendation_frontend_modality_item_embeddings = lambda **kwargs: {
            "v": frontend_v,
            "t": frontend_t,
        }

        loss = model.compute_true_missing_gcn_infonce_loss(
            torch.tensor([0, 1]),
            user_ids=torch.tensor([0, 1, 2]),
            temperature=0.2,
            bank_size=2,
            user_bank_size=3,
        )
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(frontend_v.grad)
        self.assertGreater(float(frontend_v.grad.abs().sum()), 0.0)
        self.assertIsNone(frontend_t.grad)
        self.assertIsNone(model.user_emb.weight.grad)


if __name__ == "__main__":
    unittest.main()
