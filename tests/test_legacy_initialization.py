import unittest

import torch
from torch import nn

from model import _consume_legacy_user_modality_preference_init


class LegacyInitializationTest(unittest.TestCase):
    def test_removed_embedding_keeps_published_rng_stream(self):
        seed = 2023
        num_users = 17
        num_items = 23
        embedding_dim = 5
        num_modalities = 2

        torch.manual_seed(seed)
        nn.Embedding(num_users, embedding_dim)
        nn.Embedding(num_items, embedding_dim)
        nn.Embedding(num_users, num_modalities)
        nn.Linear(embedding_dim, embedding_dim, bias=False)
        expected_user = torch.empty(num_users, embedding_dim)
        nn.init.normal_(expected_user, std=0.1)
        expected_item = torch.empty(num_items, embedding_dim)
        nn.init.normal_(expected_item, std=0.1)
        expected_rng_state = torch.random.get_rng_state()

        torch.manual_seed(seed)
        nn.Embedding(num_users, embedding_dim)
        nn.Embedding(num_items, embedding_dim)
        _consume_legacy_user_modality_preference_init(
            num_users,
            num_modalities,
            device="cpu",
            dtype=torch.float32,
        )
        nn.Linear(embedding_dim, embedding_dim, bias=False)
        actual_user = torch.empty(num_users, embedding_dim)
        nn.init.normal_(actual_user, std=0.1)
        actual_item = torch.empty(num_items, embedding_dim)
        nn.init.normal_(actual_item, std=0.1)

        torch.testing.assert_close(actual_user, expected_user, rtol=0, atol=0)
        torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng_state))


if __name__ == "__main__":
    unittest.main()
