import os
import json
import hashlib
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
from time import time
import pandas as pd

class Loader4MM(torch.utils.data.Dataset):
    def __init__(self, env):

        self.env = env
        self.split = False
        self.folds = 20
        self.n_user = 0
        self.m_item = 0


        train_file = os.path.join(self.env.DATA_PATH, 'train.txt')
        val_file = os.path.join(self.env.DATA_PATH, 'val.txt')
        test_file = os.path.join(self.env.DATA_PATH, 'test.txt')
        train_json_file = os.path.join(self.env.DATA_PATH, 'train.json')
        val_json_file = os.path.join(self.env.DATA_PATH, 'val.json')
        test_json_file = os.path.join(self.env.DATA_PATH, 'test.json')

        if not os.path.exists(train_file):
            if all(os.path.exists(path) for path in (train_json_file, val_json_file, test_json_file)):
                train_data = self.load_json_split(train_json_file)
                val_data = self.load_json_split(val_json_file)
                test_data = self.load_json_split(test_json_file)
            else:
                # Preprocessing the inter file to get train, test, and validation of users and items
                uid_field = 'userID'
                iid_field = 'itemID'
                split = 'x_label'
                cols = [uid_field, iid_field, split]

                load_inter_file = os.path.join(self.env.DATA_PATH, f"{self.env.args.dataset}.inter")

                inter_df = pd.read_csv(load_inter_file, usecols=cols, sep="\t")

                train_df = inter_df[inter_df['x_label'] == 0]
                val_df = inter_df[inter_df['x_label'] == 1]
                test_df = inter_df[inter_df['x_label'] == 2]

                train_data = self.generate_data_file(train_df, 'train')
                val_data = self.generate_data_file(val_df, 'val')
                test_data = self.generate_data_file(test_df, 'test')

            self.write_split_txt(train_file, train_data)
            self.write_split_txt(val_file, val_data)
            self.write_split_txt(test_file, test_data)

        trainUniqueUsers, trainItem, trainUser =[], [], []
        valUniqueUsers, valItem, valUser =[], [], []
        testUniqueUsers, testItem, testUser =[], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        self.train_data = defaultdict(list)
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split()
                    if not l:
                        continue
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if len(l) < 2:
                        continue
                    items = [int(i) for i in l[1:]]
                    self.train_data[uid].extend(items)
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        
      

        self.trainItem = trainItem
        setTrainItem = set(trainItem)
        self.cold_item_index = set()
        self.eval_val_missing_modality_items = self._empty_missing_metadata()
        self.val_data = defaultdict(list)
        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split()
                    if not l:
                        continue
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if len(l) < 2:
                        continue
                    else:
                        items = [int(i) for i in l[1:]]

                    for item in items:
                        if item not in setTrainItem:
                            self.cold_item_index.add(item)
                    self.val_data[uid].extend(items)
                    valUniqueUsers.append(uid)
                    valUser.extend([uid] * len(items))
                    valItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    # self.valDataSize += len(items)
        self.val_user_list = np.array(valUniqueUsers)
        self.valItem = np.array(valItem)

        self.test_data = defaultdict(list)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split()
                    if not l:
                        continue
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if len(l) < 2:
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                    for item in items:
                        if item not in setTrainItem:
                            self.cold_item_index.add(item)
                    self.test_data[uid].extend(items)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.testDataSize += len(items)
        self.cold_item_index = list(self.cold_item_index)
        self.m_item += 1
        self.n_user += 1

        self.test_user_list = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        # pre-calculate
        self._allPos = self.getUserAllItems()
        # build user-itme matrix for interaction graph data
        self.UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)), shape=(self.n_user, self.m_item))

        self.image_feat, self.text_feat, self.audio_feat = self.load_mutimedia_feature()
        self.feature = np.concatenate([self.image_feat, self.text_feat], axis=1)
        if self.audio_feat is not None:
            self.feature = np.concatenate([self.feature, self.audio_feat], axis=1)

    def _empty_missing_metadata(self):
        return {
            'items': np.array([], dtype=np.int64),
            'indicator': np.array([], dtype=np.int64),
        }

    def _dataset_seed(self):
        return int(getattr(self.env.args, 'dataset_seed', 0))

    def _sample_missing_subset(self, candidates, protected_indices, sample_size, rng=None):
        candidates = np.array(candidates, dtype=np.int64)
        if candidates.size == 0 or sample_size <= 0:
            return self._empty_missing_metadata()

        rng = rng if rng is not None else np.random.default_rng(self._dataset_seed())
        rng.shuffle(candidates)
        selected = candidates[: min(sample_size, candidates.size)]
        return {
            'items': selected,
            'indicator': protected_indices[selected],
        }

    def _sample_missing_subset_i3(self, candidates, protected_indices, sample_size, rng):
        """Match the original I3 np.random.seed/shuffle missing-mask protocol."""
        candidates = list(candidates)
        if len(candidates) == 0 or sample_size <= 0:
            return self._empty_missing_metadata()

        rng.shuffle(candidates)
        selected = np.array(candidates[: min(sample_size, len(candidates))], dtype=np.int64)
        return {
            'items': selected,
            'indicator': protected_indices[selected],
        }

    def _should_create_stage1_holdout(self):
        if float(getattr(self.env.args, 'imputation_val_rate', 0.0)) > 0:
            return True
        return getattr(self.env.args, 'stage1_profile', 'legacy') == 'v2'
            

    def generate_data_file(self, data, data_file_name):
        data_set = {}
        for column_name, item in data.iterrows():
            user_id = item['userID']
            item_id = item['itemID']
            if user_id in data_set:
                data_set[user_id].append(item_id)
            else:
                data_set[user_id] = [item_id]
        
        return data_set

    def load_json_split(self, split_file):
        data = pd.read_json(split_file, typ='series')
        split_dict = {}
        for user_id, item_ids in data.items():
            split_dict[int(user_id)] = [int(item_id) for item_id in item_ids]
        return split_dict

    def write_split_txt(self, output_file, split_data):
        with open(output_file, encoding='utf-8', mode='w') as f:
            for user in sorted(split_data.keys()):
                items = split_data[user]
                if items:
                    line = str(user) + ' ' + ' '.join(map(str, items))
                else:
                    line = str(user)
                f.write(line + '\n')

    def _adj_cache_paths(self):
        cache_path = os.path.join(self.env.DATA_PATH, 's_pre_adj_mat.npz')
        meta_path = os.path.join(self.env.DATA_PATH, 's_pre_adj_mat.meta.json')
        return cache_path, meta_path

    def _train_graph_fingerprint(self):
        graph = self.UserItemNet.tocsr()
        digest = hashlib.sha256()
        digest.update(np.asarray(graph.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(graph.indptr, dtype=np.int64).tobytes())
        digest.update(np.asarray(graph.indices, dtype=np.int64).tobytes())
        return {
            'version': 1,
            'dataset': self.env.args.dataset,
            'n_user': int(self.n_user),
            'm_item': int(self.m_item),
            'train_interactions': int(self.traindataSize),
            'user_item_nnz': int(graph.nnz),
            'adj_shape': [int(self.n_user + self.m_item), int(self.n_user + self.m_item)],
            'adj_nnz': int(graph.nnz * 2),
            'fingerprint': digest.hexdigest(),
        }

    def _load_valid_adj_cache(self):
        cache_path, meta_path = self._adj_cache_paths()
        expected = self._train_graph_fingerprint()
        if not os.path.exists(cache_path):
            return None
        if not os.path.exists(meta_path):
            print('ignore adjacency cache without metadata; rebuilding train-only graph')
            return None

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                cached_meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            print('ignore adjacency cache with unreadable metadata; rebuilding train-only graph')
            return None

        for key, value in expected.items():
            if cached_meta.get(key) != value:
                print(f'ignore stale adjacency cache: metadata mismatch on {key}; rebuilding train-only graph')
                return None

        try:
            cached_adj = sp.load_npz(cache_path)
        except (OSError, ValueError):
            print('ignore unreadable adjacency cache; rebuilding train-only graph')
            return None

        if list(cached_adj.shape) != expected['adj_shape'] or int(cached_adj.nnz) != expected['adj_nnz']:
            print('ignore adjacency cache with invalid shape/nnz; rebuilding train-only graph')
            return None

        print('successfully loaded verified train-only adjacency cache')
        return cached_adj

    def _build_train_only_norm_adj(self):
        print("generating adjacency matrix")
        s = time()
        n_user = self.n_user
        m_item = self.m_item
        adj_mat = sp.dok_matrix((n_user + m_item, n_user + m_item), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()

        print(adj_mat.shape, adj_mat[:n_user, n_user:].shape, R.shape)

        adj_mat[:n_user, n_user:] = R
        adj_mat[n_user:, :n_user] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time()
        print(f"costing {end-s}s, saved norm_mat...")

        cache_path, meta_path = self._adj_cache_paths()
        sp.save_npz(cache_path, norm_adj)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self._train_graph_fingerprint(), f, indent=2, sort_keys=True)
        return norm_adj
 
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            norm_adj = self._load_valid_adj_cache()
            if norm_adj is None:
                norm_adj = self._build_train_only_norm_adj()

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.env.device)
                print("don't split the matrix")
        return self.Graph

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_user + self.m_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_user + self.m_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.env.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _normalize_item_graph(self, graph, norm_type):
        graph = graph.tocsr().astype(np.float32)
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        if norm_type == 'none':
            return graph

        rowsum = np.asarray(graph.sum(axis=1)).flatten()
        if norm_type == 'rw':
            d_inv = np.zeros_like(rowsum, dtype=np.float32)
            nonzero = rowsum > 0
            d_inv[nonzero] = 1.0 / rowsum[nonzero]
            return sp.diags(d_inv).dot(graph).tocsr()
        if norm_type == 'sym':
            d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
            nonzero = rowsum > 0
            d_inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
            d_mat = sp.diags(d_inv_sqrt)
            return d_mat.dot(graph).dot(d_mat).tocsr()
        raise ValueError(f'Unsupported item graph normalization: {norm_type}')

    def _topk_sparse_rows(self, matrix, topk):
        matrix = matrix.tocsr().astype(np.float32)
        matrix.setdiag(0.0)
        matrix.eliminate_zeros()
        if topk <= 0:
            return matrix

        rows, cols, data = [], [], []
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            row_cols = matrix.indices[start:end]
            row_data = matrix.data[start:end]
            if row_data.size == 0:
                continue
            if row_data.size > topk:
                keep = np.argpartition(row_data, -topk)[-topk:]
                keep = keep[np.argsort(row_data[keep])[::-1]]
                row_cols = row_cols[keep]
                row_data = row_data[keep]
            rows.extend([row] * row_data.size)
            cols.extend(row_cols.tolist())
            data.extend(row_data.tolist())
        return sp.csr_matrix((data, (rows, cols)), shape=matrix.shape, dtype=np.float32)

    def _scale_cf_item_graph(self, graph, scale, power=0.5, clip=3.0):
        scale = str(scale or 'raw')
        graph = graph.tocsr().astype(np.float32)
        if scale == 'raw':
            return graph
        if scale == 'sqrt':
            graph = graph.copy()
            graph.data = np.sqrt(graph.data).astype(np.float32)
            return graph
        if scale == 'power':
            graph = graph.copy()
            graph.data = np.power(graph.data, float(power)).astype(np.float32)
            return graph
        if scale == 'clip':
            graph = graph.copy()
            graph.data = np.minimum(graph.data, float(clip)).astype(np.float32)
            return graph
        if scale == 'cosine':
            item_pop = np.asarray(self.UserItemNet.sum(axis=0)).flatten().astype(np.float32)
            inv_sqrt = np.zeros_like(item_pop, dtype=np.float32)
            nonzero = item_pop > 0
            inv_sqrt[nonzero] = np.power(item_pop[nonzero], -0.5)
            d_mat = sp.diags(inv_sqrt)
            return d_mat.dot(graph).dot(d_mat).tocsr().astype(np.float32)
        if scale == 'log1p':
            graph = graph.copy()
            graph.data = np.log1p(graph.data).astype(np.float32)
            return graph
        if scale in ('rowmax', 'log1p_rowmax'):
            graph = graph.copy()
            if scale == 'log1p_rowmax':
                graph.data = np.log1p(graph.data).astype(np.float32)
            rowsum_max = np.zeros(graph.shape[0], dtype=np.float32)
            for row in range(graph.shape[0]):
                start, end = graph.indptr[row], graph.indptr[row + 1]
                if end > start:
                    rowsum_max[row] = graph.data[start:end].max()
            inv = np.zeros_like(rowsum_max, dtype=np.float32)
            nonzero = rowsum_max > 0
            inv[nonzero] = 1.0 / rowsum_max[nonzero]
            return sp.diags(inv).dot(graph).tocsr().astype(np.float32)
        raise ValueError(f'Unsupported CF item graph scale: {scale}')

    def _build_cf_item_graph(self, topk, scale='raw', power=0.5, clip=3.0):
        graph = self.UserItemNet.T.dot(self.UserItemNet).tocsr().astype(np.float32)
        graph = self._scale_cf_item_graph(graph, scale, power=power, clip=clip)
        return self._topk_sparse_rows(graph, topk)

    def _build_feature_item_graph(self, feature, topk, chunk_size):
        feature = np.asarray(feature, dtype=np.float32)
        norms = np.linalg.norm(feature, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feature = feature / norms

        n_item = feature.shape[0]
        chunk_size = max(int(chunk_size), 1)
        k = min(max(int(topk), 1) + 1, n_item)
        rows, cols, data = [], [], []
        for start in tqdm(range(0, n_item, chunk_size), desc='building feature item graph'):
            end = min(start + chunk_size, n_item)
            sim = np.matmul(feature[start:end], feature.T)
            local_rows = np.arange(start, end)
            sim[np.arange(end - start), local_rows] = -np.inf
            top_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
            top_val = np.take_along_axis(sim, top_idx, axis=1)
            order = np.argsort(top_val, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(top_idx, order, axis=1)
            top_val = np.take_along_axis(top_val, order, axis=1)
            for offset, row in enumerate(local_rows):
                valid = np.isfinite(top_val[offset]) & (top_val[offset] > 0)
                row_cols = top_idx[offset][valid][:topk]
                row_data = top_val[offset][valid][:topk]
                if row_data.size == 0:
                    continue
                rows.extend([int(row)] * row_data.size)
                cols.extend(row_cols.astype(np.int64).tolist())
                data.extend(row_data.astype(np.float32).tolist())
        return sp.csr_matrix((data, (rows, cols)), shape=(n_item, n_item), dtype=np.float32)

    @property
    def allPos(self):
        return self._allPos

    def set_miss_mutimedia_feature_items(self, fea, seed=None, rate=0.3, exp_mode='fm', path=''):

        self.train_missing_modality_items = {}
        self.val_missing_modality_items = self._empty_missing_metadata()
        self.test_missing_modality_items = {}

        seed = self._dataset_seed() if seed is None else seed
        use_i3_missing_protocol = getattr(self.env.args, 'missing_mask_protocol', 'i3') == 'i3'
        rng = np.random.RandomState(seed) if use_i3_missing_protocol else np.random.default_rng(seed)
        n_modality = len(fea)
        n_item = fea[0].shape[0]
        # random missing modality index
        if use_i3_missing_protocol:
            protected_indices = rng.randint(n_modality, size=n_item)
        else:
            protected_indices = rng.integers(n_modality, size=n_item)
        self.protected_indices = protected_indices

        # I3 samples test first and train second from the same legacy RNG stream.
        # Keep that order so seed=0 produces the same train/test missing masks.
        eval_missing_rate = float(getattr(self.env.args, 'eval_missing_rate', 0.5))
        if eval_missing_rate < 0.0 or eval_missing_rate > 1.0:
            raise ValueError(f'eval_missing_rate must be in [0, 1], got {eval_missing_rate}')

        test_candidate_data = list(set(self.testItem))
        test_num_missing_entries = int(len(test_candidate_data) * eval_missing_rate)
        if use_i3_missing_protocol:
            self.test_missing_modality_items = self._sample_missing_subset_i3(
                test_candidate_data,
                protected_indices,
                test_num_missing_entries,
                rng,
            )
        else:
            self.test_missing_modality_items = self._sample_missing_subset(
                test_candidate_data,
                protected_indices,
                test_num_missing_entries,
                rng=rng,
            )

        val_candidate_data = list(set(self.valItem))
        val_num_missing_entries = int(len(val_candidate_data) * eval_missing_rate)
        val_rng = np.random.default_rng(seed + 1000003) if use_i3_missing_protocol else rng
        self.eval_val_missing_modality_items = self._sample_missing_subset(
            val_candidate_data,
            protected_indices,
            val_num_missing_entries,
            rng=val_rng,
        )

        train_candidate_data = list(set(self.trainItem))
        if self._should_create_stage1_holdout():
            val_rate = max(0.0, float(getattr(self.env.args, 'imputation_val_rate', 0.0)))
            train_candidates = np.array(train_candidate_data, dtype=np.int64)
            rng.shuffle(train_candidates)

            val_num_missing_entries = 0
            if val_rate > 0 and train_candidates.size > 0:
                val_num_missing_entries = min(
                    train_candidates.size,
                    max(1, int(train_candidates.size * val_rate)),
                )
            val_candidates = train_candidates[:val_num_missing_entries]
            remaining_train_candidates = train_candidates[val_num_missing_entries:]

            self.val_missing_modality_items = self._sample_missing_subset(
                val_candidates,
                protected_indices,
                len(val_candidates),
                rng=rng,
            )
            train_num_missing_entries = int(len(remaining_train_candidates) * rate)
            self.train_missing_modality_items = self._sample_missing_subset(
                remaining_train_candidates,
                protected_indices,
                train_num_missing_entries,
                rng=rng,
            )
            self.stage1_train_items = remaining_train_candidates.astype(np.int64)
            print(
                'sample items with missing modality successfuly, '
                'seed = {0}, train dataset include {1} items, imputation val include {2} items, '
                'test dataset include {3} items'.format(
                    seed,
                    len(self.train_missing_modality_items['items']),
                    len(self.val_missing_modality_items['items']),
                    len(self.test_missing_modality_items['items']),
                )
            )
        else:
            train_candidates = np.array(train_candidate_data, dtype=np.int64)
            train_num_missing_entries = int(len(train_candidate_data) * rate)
            if use_i3_missing_protocol:
                self.train_missing_modality_items = self._sample_missing_subset_i3(
                    train_candidate_data,
                    protected_indices,
                    train_num_missing_entries,
                    rng,
                )
            else:
                self.train_missing_modality_items = self._sample_missing_subset(
                    train_candidate_data,
                    protected_indices,
                    train_num_missing_entries,
                    rng=rng,
                )
            self.stage1_train_items = train_candidates.astype(np.int64)
            print(
                'sample items with missing modality successfuly, seed = {0}, train dataset include {1} items, test dataset include {2} items'.format(
                    seed,
                    len(self.train_missing_modality_items['items']),
                    len(self.test_missing_modality_items['items']),
                )
            )

    def sample_stage1_dynamic_missing_metadata(self, seed=None, rate=None):
        protected_indices = getattr(self, 'protected_indices', None)
        if protected_indices is None:
            raise RuntimeError('protected_indices are not initialized')

        candidates = np.array(
            self.stage1_train_items if self.stage1_train_items is not None else list(set(self.trainItem)),
            dtype=np.int64,
        )
        if candidates.size == 0:
            return self._empty_missing_metadata()

        sample_rate = self.env.args.missing_rate if rate is None else rate
        sample_size = int(candidates.size * sample_rate)
        if sample_rate > 0 and candidates.size > 0:
            sample_size = max(1, sample_size)

        rng = np.random.default_rng(seed if seed is not None else self._dataset_seed())
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        selected = shuffled[: min(sample_size, shuffled.size)]
        if selected.size == 0:
            return self._empty_missing_metadata()

        return {
            'items': selected.astype(np.int64),
            'indicator': protected_indices[selected].astype(np.int64),
        }

    def refresh_stage1_dynamic_train_missing_metadata(self, seed=None, rate=None):
        self.train_missing_modality_items = self.sample_stage1_dynamic_missing_metadata(seed=seed, rate=rate)
        return self.train_missing_modality_items


    def _resolve_modal_feature_dir(self, explicit_dir, base_dir, phase_name):
        if explicit_dir:
            return os.path.expanduser(explicit_dir)
        phase_dir = os.path.join(base_dir, phase_name)
        if os.path.isdir(phase_dir):
            return phase_dir
        return base_dir

    def _load_modal_feature_pair(self, feature_dir, label):
        image_name = getattr(self.env.args, 'modal_feature_image_file', 'agg_image_items.npy')
        text_name = getattr(self.env.args, 'modal_feature_text_file', 'agg_text_items.npy')
        image_file = os.path.join(feature_dir, image_name)
        text_file = os.path.join(feature_dir, text_name)
        if not os.path.exists(image_file):
            raise FileNotFoundError(f'{label} image feature file not found: {image_file}')
        if not os.path.exists(text_file):
            raise FileNotFoundError(f'{label} text feature file not found: {text_file}')

        image_feat = np.load(image_file).astype(np.float32, copy=False)
        text_feat = np.load(text_file).astype(np.float32, copy=False)
        for name, feature in (('image', image_feat), ('text', text_feat)):
            if feature.shape[0] != self.m_item:
                raise ValueError(
                    f'{label} {name} feature item count mismatch: '
                    f'feature has {feature.shape[0]} rows, dataset has {self.m_item} items'
                )
        return image_feat, text_feat

    def _load_modal_observed_mask_pair(self, feature_dir, label):
        image_name = getattr(self.env.args, 'modal_feature_image_mask_file', 'image_observed_mask.npy')
        text_name = getattr(self.env.args, 'modal_feature_text_mask_file', 'text_observed_mask.npy')
        image_file = os.path.join(feature_dir, image_name)
        text_file = os.path.join(feature_dir, text_name)
        if not os.path.exists(image_file) or not os.path.exists(text_file):
            return None

        image_mask = np.load(image_file).astype(bool)
        text_mask = np.load(text_file).astype(bool)
        for name, mask in (('image', image_mask), ('text', text_mask)):
            if mask.shape[0] != self.m_item:
                raise ValueError(
                    f'{label} {name} observed mask item count mismatch: '
                    f'mask has {mask.shape[0]} rows, dataset has {self.m_item} items'
                )
        return {'v': image_mask, 't': text_mask}

    def _load_modal_feature_override(self):
        override_dir = getattr(self.env.args, 'modal_feature_override_dir', '') or ''
        if not override_dir:
            return None

        base_dir = os.path.expanduser(override_dir)
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f'modal_feature_override_dir not found: {base_dir}')

        train_dir = self._resolve_modal_feature_dir(
            getattr(self.env.args, 'modal_feature_train_dir', '') or '',
            base_dir,
            'phase_train',
        )
        eval_dir = self._resolve_modal_feature_dir(
            getattr(self.env.args, 'modal_feature_eval_dir', '') or '',
            base_dir,
            'phase_eval',
        )
        image_feat, text_feat = self._load_modal_feature_pair(train_dir, 'train')
        eval_image_feat, eval_text_feat = self._load_modal_feature_pair(eval_dir, 'eval')
        self.train_external_modal_observed_masks = self._load_modal_observed_mask_pair(train_dir, 'train')
        self.eval_external_modal_observed_masks = self._load_modal_observed_mask_pair(eval_dir, 'eval')

        self.eval_image_feat = eval_image_feat
        self.eval_text_feat = eval_text_feat
        self.uses_split_modal_feature_override = os.path.abspath(train_dir) != os.path.abspath(eval_dir)
        print(
            'loaded external modal features: '
            f'train_dir={train_dir}, eval_dir={eval_dir}, '
            f'image_shape={image_feat.shape}, text_shape={text_feat.shape}'
        )
        if self.train_external_modal_observed_masks is not None:
            print(
                'loaded external modal observed masks: '
                f'train_v={int(self.train_external_modal_observed_masks["v"].sum())}, '
                f'train_t={int(self.train_external_modal_observed_masks["t"].sum())}, '
                f'eval_v={int(self.eval_external_modal_observed_masks["v"].sum())}, '
                f'eval_t={int(self.eval_external_modal_observed_masks["t"].sum())}'
            )
        return image_feat, text_feat, None

    def load_mutimedia_feature(self):
        override_features = self._load_modal_feature_override()
        if override_features is not None:
            image_feat, text_feat, audio_feat = override_features
            fea = [image_feat, text_feat]
        else:
            image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
            text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')
            audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat.npy')
            image_feat = np.load(image_file)
            text_feat = np.load(text_file)
            fea = [image_feat, text_feat]

            audio_feat = None
            if os.path.exists(audio_file):
                audio_feat = np.load(audio_file)
                fea += [audio_feat]

            self.eval_image_feat = image_feat
            self.eval_text_feat = text_feat
            self.uses_split_modal_feature_override = False

        if audio_feat is None:
            self.eval_audio_feat = None
        else:
            self.eval_audio_feat = audio_feat

        self.set_miss_mutimedia_feature_items(
            fea,
            seed=self._dataset_seed(),
            rate=self.env.args.missing_rate,
            exp_mode=self.env.args.exp_mode,
        )

        if audio_feat is not None:
            return image_feat, text_feat, audio_feat
        else:
            return image_feat, text_feat, None


    def getUserAllItems(self):
        posItems = defaultdict(list)

        for user in list(self.train_data.keys()):
            posItems[user].extend(self.train_data[user])
        return posItems

    def get_stage1_train_items(self):
        if self.stage1_train_items is not None:
            return np.array(self.stage1_train_items, dtype=np.int64)
        return np.array(list(set(self.trainItem)), dtype=np.int64)

    def neg_sample(self):
        self.pair_data = []
        print('generate samples ...')
        for user_id in tqdm(self.train_data.keys()):
            positive_list = self.train_data[user_id]  # self.train_dict[user_id]
            for item_i in positive_list:
                item_j = np.random.randint(self.m_item)
                while item_j in positive_list:
                    item_j = np.random.randint(self.m_item)
                self.pair_data.append([user_id, item_i, item_j])

    def neg_uniform_sample(self):
        user_num = len(self.trainUser)
        users = np.random.randint(0, self.n_user, user_num)
        self.pair_data = []
        print('generate uniform samples ...')
        for user_id in tqdm(users):
            positive_list = self.train_data[user_id]  # self.train_dict[user_id]
            if len(positive_list) == 0:
                continue
            posindex = np.random.randint(0, len(positive_list))
            item_i = positive_list[posindex]
            item_j = np.random.randint(self.m_item)
            while item_j in positive_list:
                item_j = np.random.randint(self.m_item)
            self.pair_data.append([user_id, item_i, item_j])

    def __getitem__(self, index):
        user = self.pair_data[index][0]
        pos_item = self.pair_data[index][1]
        neg_item = self.pair_data[index][2]
        return user, pos_item, neg_item

    def __len__(self):
        return len(self.trainItem)

def Uniform_PairSample(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = len(dataset.trainItem)
    users = np.random.randint(0, dataset.n_user, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_item)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def PairSample(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(dataset.train_data.keys()):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        for positem in dataset.train_data[user]:
            while True:
                negitem = np.random.randint(0, dataset.m_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            random_user = np.random.randint(0, dataset.n_user)
            random_item = np.random.randint(0, dataset.m_item)
            S.append([user, positem, negitem, random_user, random_item])
    return np.array(S)


def ItemSample(dataset):
    """
    Unique training items for completion pretraining.
    This keeps stage1 focused on item-level completion rather than
    user-item pairwise recommendation batches.
    """
    if hasattr(dataset, 'get_stage1_train_items'):
        return dataset.get_stage1_train_items()
    return np.array(list(set(dataset.trainItem)), dtype=np.int64)

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
