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
        split_path = self.env.DATA_PATH

        train_file = os.path.join(split_path, 'train.txt')
        val_file = os.path.join(split_path, 'val.txt')
        test_file = os.path.join(split_path, 'test.txt')
        self.interaction_split_path = split_path
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
                    self.test_data[uid].extend(items)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1

        self.test_user_list = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.negative_item_pool = np.arange(self.m_item, dtype=np.int64)

        self.Graph = None
        # pre-calculate
        self._allPos = self.getUserAllItems()
        # build user-itme matrix for interaction graph data
        self.UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)), shape=(self.n_user, self.m_item))

        self.image_feat, self.text_feat, self.audio_feat, self.video_feat = self.load_mutimedia_feature()
        self.feature = np.concatenate([self.image_feat, self.text_feat], axis=1)
        if self.audio_feat is not None:
            self.feature = np.concatenate([self.feature, self.audio_feat], axis=1)
        if self.video_feat is not None:
            self.feature = np.concatenate([self.feature, self.video_feat], axis=1)

    def _empty_missing_metadata(self):
        return {
            'items': np.array([], dtype=np.int64),
            'indicator': np.array([], dtype=np.int64),
        }

    def _missing_indicator_counts(self, metadata, n_modality):
        indicators = np.asarray(metadata['indicator'], dtype=np.int64)
        return [int(np.count_nonzero(indicators == index)) for index in range(n_modality)]

    def _log_missing_protocol(self, n_modality, rate):
        names = ['image', 'text', 'audio', 'video'][:n_modality]
        split_metadata = (
            ('train', self.train_missing_modality_items),
            ('stage1_val', self.val_missing_modality_items),
            ('val', self.eval_val_missing_modality_items),
            ('test', self.test_missing_modality_items),
        )
        split_counts = []
        for split, metadata in split_metadata:
            counts = self._missing_indicator_counts(metadata, n_modality)
            modal_counts = ','.join(f'{name}:{count}' for name, count in zip(names, counts))
            split_counts.append(f'{split}={len(metadata["items"])}[{modal_counts}]')
        print(
            f'phase-invariant missing payload: rate={rate}; '
            + '; '.join(split_counts)
        )
            

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
        cache_root = getattr(self, 'interaction_split_path', self.env.DATA_PATH)
        cache_path = os.path.join(cache_root, 's_pre_adj_mat.npz')
        meta_path = os.path.join(cache_root, 's_pre_adj_mat.meta.json')
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

    def _build_cf_item_similarity(self, scale='raw', power=0.5, clip=3.0):
        graph = self.UserItemNet.T.dot(self.UserItemNet).tocsr().astype(np.float32)
        return self._scale_cf_item_graph(graph, scale, power=power, clip=clip)

    def _build_cf_item_graph(self, topk, scale='raw', power=0.5, clip=3.0):
        graph = self._build_cf_item_similarity(
            scale=scale,
            power=power,
            clip=clip,
        )
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

    def _build_fused_cf_feature_item_graph(
        self,
        feature,
        cf_graph,
        topk,
        chunk_size,
        cf_weight,
        feature_weight,
    ):
        """Fuse full CF and semantic scores, then select one final row-wise top-k."""
        feature = np.asarray(feature, dtype=np.float32)
        norms = np.linalg.norm(feature, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feature = feature / norms

        n_item = feature.shape[0]
        cf_graph = cf_graph.tocsr().astype(np.float32)
        if cf_graph.shape != (n_item, n_item):
            raise ValueError(
                f'CF graph shape must match item features: {cf_graph.shape} != {(n_item, n_item)}'
            )

        cf_weight = max(float(cf_weight), 0.0)
        feature_weight = max(float(feature_weight), 0.0)
        total_weight = cf_weight + feature_weight
        if total_weight <= 0.0:
            raise ValueError('Fused CF-feature graph requires a positive source weight')

        chunk_size = max(int(chunk_size), 1)
        topk = min(max(int(topk), 1), max(n_item - 1, 1))
        candidate_k = min(topk + 1, n_item)
        rows, cols, data = [], [], []
        for start in tqdm(range(0, n_item, chunk_size), desc='building fuse-before-topk item graph'):
            end = min(start + chunk_size, n_item)
            scores = np.matmul(feature[start:end], feature.T)
            scores *= feature_weight
            if cf_weight > 0.0:
                scores += cf_weight * cf_graph[start:end].toarray()
            scores *= 1.0 / total_weight

            local_rows = np.arange(start, end)
            scores[np.arange(end - start), local_rows] = -np.inf
            top_idx = np.argpartition(scores, -candidate_k, axis=1)[:, -candidate_k:]
            top_val = np.take_along_axis(scores, top_idx, axis=1)
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

    def load_missing_payload(self, features, rate):
        dataset = self.env.args.dataset
        filename = f'unified_missing_items_mr{float(rate):g}_seed2023.npy'
        payload_file = os.path.join(self.env.DATA_PATH, filename)
        if not os.path.isfile(payload_file):
            payload_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'configs',
                dataset,
                filename,
            )
        if not os.path.isfile(payload_file):
            raise FileNotFoundError(f'unified missing payload not found: {payload_file}')

        payload = np.load(payload_file, allow_pickle=True).item()
        if payload.get('protocol') != 'unified_single_modality':
            raise ValueError(f'unsupported unified payload protocol in {payload_file}')
        if payload.get('dataset') != dataset:
            raise ValueError(f'unified payload dataset mismatch in {payload_file}')
        if not np.isclose(float(payload.get('missing_rate')), float(rate)):
            raise ValueError(f'unified payload rate mismatch in {payload_file}')

        items = np.asarray(payload['items'], dtype=np.int64)
        indicators = np.asarray(payload['indicator'], dtype=np.int64)
        n_items = features[0].shape[0]
        n_modalities = len(features)
        if items.ndim != 1 or indicators.shape != items.shape:
            raise ValueError(f'invalid items/indicator arrays in {payload_file}')
        if (
            len(np.unique(items)) != len(items)
            or np.any(items < 0)
            or np.any(items >= n_items)
        ):
            raise ValueError(f'invalid or duplicate item ids in {payload_file}')
        if np.any(indicators < 0) or np.any(indicators >= n_modalities):
            raise ValueError(f'invalid modality indicator in {payload_file}')

        metadata = {'items': items.copy(), 'indicator': indicators.copy()}
        self.train_missing_modality_items = {
            key: value.copy() for key, value in metadata.items()
        }
        self.eval_val_missing_modality_items = {
            key: value.copy() for key, value in metadata.items()
        }
        self.test_missing_modality_items = {
            key: value.copy() for key, value in metadata.items()
        }
        self.val_missing_modality_items = self._empty_missing_metadata()
        self.stage1_train_items = np.unique(np.asarray(self.trainItem, dtype=np.int64))
        self.protected_indices = np.zeros(n_items, dtype=np.int64)
        self.protected_indices[items] = indicators
        self.train_protected_indices = self.protected_indices.copy()
        print(f'loaded phase-invariant unified missing payload: {payload_file}')
        self._log_missing_protocol(n_modalities, rate)


    def load_mutimedia_feature(self):
        image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
        text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')
        audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat.npy')
        video_file = os.path.join(self.env.DATA_PATH, 'video_feat.npy')
        image_feat = np.load(image_file)
        text_feat = np.load(text_file)
        fea = [image_feat, text_feat]

        audio_feat = None
        if os.path.exists(audio_file):
            audio_feat = np.load(audio_file)
            fea.append(audio_feat)

        video_feat = None
        if os.path.exists(video_file):
            video_feat = np.load(video_file)
            fea.append(video_feat)

        self.load_missing_payload(fea, rate=self.env.args.missing_rate)
        return image_feat, text_feat, audio_feat, video_feat

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
                item_j = _sample_negative_item(self, positive_list)
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
            item_j = _sample_negative_item(self, positive_list)
            self.pair_data.append([user_id, item_i, item_j])

    def __getitem__(self, index):
        user = self.pair_data[index][0]
        pos_item = self.pair_data[index][1]
        neg_item = self.pair_data[index][2]
        return user, pos_item, neg_item

    def __len__(self):
        return len(self.trainItem)

def _sample_negative_item(dataset, positive_items):
    pool = np.asarray(getattr(dataset, 'negative_item_pool', np.arange(dataset.m_item)), dtype=np.int64)
    if pool.size == 0:
        raise ValueError('Negative item pool is empty')
    positive_set = positive_items if isinstance(positive_items, set) else set(positive_items)
    if len(positive_set) >= pool.size and all(int(item) in positive_set for item in pool):
        raise ValueError('User has no available negative item in the configured pool')
    while True:
        candidate = int(pool[np.random.randint(pool.size)])
        if candidate not in positive_set:
            return candidate


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
        negitem = _sample_negative_item(dataset, posForUser)
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
            negitem = _sample_negative_item(dataset, posForUser)
            random_user = np.random.randint(0, dataset.n_user)
            pool = np.asarray(getattr(dataset, 'negative_item_pool', np.arange(dataset.m_item)))
            random_item = int(pool[np.random.randint(pool.size)])
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
