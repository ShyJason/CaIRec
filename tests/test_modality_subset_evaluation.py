import unittest

import numpy as np

import evaluation


class ModalitySubsetEvaluationTest(unittest.TestCase):
    def test_split_is_interaction_level_and_drops_empty_users(self):
        ratings = {
            0: [1, 2],
            1: [2, 3],
            2: [4],
        }

        missing, full = evaluation.split_ratings_by_item_membership(
            ratings, np.asarray([2, 4])
        )

        self.assertEqual(missing, {0: [2], 1: [2], 2: [4]})
        self.assertEqual(full, {0: [1], 1: [3]})
        self.assertEqual(
            evaluation.rating_counts(missing),
            {'users': 3, 'interactions': 3},
        )
        self.assertEqual(
            evaluation.rating_counts(full),
            {'users': 2, 'interactions': 2},
        )

    def test_subset_metrics_keep_the_full_candidate_ranking(self):
        users = np.asarray([[1.0, 0.0]], dtype=np.float32)
        items = np.asarray(
            [
                [0.0, 0.0],
                [3.0, 0.0],  # full-modal positive, rank 1
                [2.0, 0.0],  # missing-modal positive, rank 2
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        ratings = {0: [1, 2]}
        missing, full = evaluation.split_ratings_by_item_membership(ratings, [2])

        _, missing_recall, _ = evaluation.num_faiss_evaluate(
            missing, [0], {0: []}, [1, 2], users, items
        )
        _, full_recall, _ = evaluation.num_faiss_evaluate(
            full, [0], {0: []}, [1, 2], users, items
        )

        self.assertEqual(missing_recall[1], 0.0)
        self.assertEqual(missing_recall[2], 1.0)
        self.assertEqual(full_recall[1], 1.0)
        self.assertEqual(full_recall[2], 1.0)


if __name__ == '__main__':
    unittest.main()
