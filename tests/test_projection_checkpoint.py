import tempfile
import unittest
from pathlib import Path

import torch

from model import MILK_model


ROOT = Path(__file__).resolve().parents[1]


class ProjectionCheckpointTest(unittest.TestCase):
    def _model(self):
        model = MILK_model.__new__(MILK_model)
        torch.nn.Module.__init__(model)
        model.modalities = ["v", "t"]
        model.use_decoupled_latent_bridge = True
        model.use_latent_direct_bridge = False
        model.comp_proj_v = torch.nn.Linear(3, 2)
        model.comp_proj_t = torch.nn.Linear(4, 2)
        model.W = torch.nn.ParameterDict(
            {"v": torch.nn.Parameter(torch.zeros(2, 2))}
        )
        return model

    def test_loads_only_projection_tensors(self):
        model = self._model()
        original_w = model.W["v"].detach().clone()
        checkpoint = {
            "model_state_dict": {
                "comp_proj_v.weight": torch.full_like(model.comp_proj_v.weight, 2.0),
                "comp_proj_v.bias": torch.full_like(model.comp_proj_v.bias, 3.0),
                "comp_proj_t.weight": torch.full_like(model.comp_proj_t.weight, 4.0),
                "comp_proj_t.bias": torch.full_like(model.comp_proj_t.bias, 5.0),
                "W.v": torch.ones_like(model.W["v"]),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.pth"
            torch.save(checkpoint, path)
            matched = model.load_projection_checkpoint(path)

        self.assertEqual(
            matched,
            [
                "comp_proj_t.bias",
                "comp_proj_t.weight",
                "comp_proj_v.bias",
                "comp_proj_v.weight",
            ],
        )
        self.assertTrue(torch.equal(model.comp_proj_v.weight, torch.full_like(model.comp_proj_v.weight, 2.0)))
        self.assertTrue(torch.equal(model.comp_proj_t.bias, torch.full_like(model.comp_proj_t.bias, 5.0)))
        self.assertTrue(torch.equal(model.W["v"], original_w))

    def test_rejects_incomplete_projection_checkpoint(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            torch.save(
                {
                    "model_state_dict": {
                        "comp_proj_v.weight": torch.ones_like(model.comp_proj_v.weight),
                        "comp_proj_v.bias": torch.ones_like(model.comp_proj_v.bias),
                    }
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "missing tensors=.*comp_proj_t"):
                model.load_projection_checkpoint(path)

    def test_bundled_projections_are_safe_tensor_checkpoints(self):
        expected_keys = {
            "comp_proj_v.weight",
            "comp_proj_v.bias",
            "comp_proj_t.weight",
            "comp_proj_t.bias",
        }
        for dataset in ("clothing", "beauty", "sports"):
            with self.subTest(dataset=dataset):
                checkpoint = torch.load(
                    ROOT / "ckpt" / f"{dataset}.pth",
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(
                    set(checkpoint["model_state_dict"]),
                    expected_keys,
                )


if __name__ == "__main__":
    unittest.main()
