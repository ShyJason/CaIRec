import os
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

        if not os.path.exists(train_file):

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


            train_df_item_set = list(set(train_df['itemID'].unique()))
            val_df_item_set = list(set(val_df['itemID'].unique()))
            test_df_item_set = list(set(test_df['itemID'].unique()))


            train_data = self.generate_data_file(train_df, 'train')
            val_data = self.generate_data_file(val_df, 'val')
            test_data = self.generate_data_file(test_df, 'test')

            with open(train_file, encoding='utf-8', mode='w') as f:
                for user in list(train_data.keys()):
                    s = str(user) + ' '
                    s = s + ' '.join(list(map(lambda x:str(x), train_data[user])))
                    f.write(s+'\n')
            with open(val_file, encoding='utf-8', mode='w') as f:
                for user in list(val_data.keys()):
                    s = str(user) + ' '
                    s = s + ' '.join(list(map(lambda x:str(x), val_data[user])))
                    f.write(s+'\n')
            with open(test_file, encoding='utf-8', mode='w') as f:
                for user in list(test_data.keys()):
                    s = str(user) + ' '
                    s = s + ' '.join(list(map(lambda x:str(x), test_data[user])))
                    f.write(s+'\n')

        trainUniqueUsers, trainItem, trainUser =[], [], []
        valUniqueUsers, valItem, valUser =[], [], []
        testUniqueUsers, testItem, testUser =[], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        self.train_data = defaultdict(list)
        with open(train_file) as f:
            for l in f.readlines():
                l = l.strip().split()
                if len(l) == 0:
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
                self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        
      

        self.trainItem = trainItem
        setTrainItem = set(trainItem)
        self.cold_item_index = set()
        self.val_data = defaultdict(list)
        with open(val_file) as f:
            for l in f.readlines():
                l = l.strip().split()
                if len(l) == 0:
                    continue
                uid = int(l[0])
                self.n_user = max(self.n_user, uid)
                if len(l) < 2:
                    continue
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

        self.test_data = defaultdict(list)
        with open(test_file) as f:
            for l in f.readlines():
                l = l.strip().split()
                if len(l) == 0:
                    continue
                uid = int(l[0])
                self.n_user = max(self.n_user, uid)
                if len(l) < 2:
                    continue
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
 
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(os.path.join(self.env.DATA_PATH, 's_pre_adj_mat.npz'))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                n_user = self.n_user
                m_item = self.m_item
                # self.n_user+1, self.m_item+1
                adj_mat = sp.dok_matrix((n_user + m_item, n_user + m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()

                print(adj_mat.shape,adj_mat[:n_user, n_user:].shape, R.shape)

                adj_mat[:n_user, n_user:] = R
                adj_mat[n_user:, :n_user] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(os.path.join(self.env.DATA_PATH, 's_pre_adj_mat.npz'), norm_adj)

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


    @property
    def allPos(self):
        return self._allPos

    def set_miss_mutimedia_feature_items(self, fea, seed=0, rate=0.3, exp_mode='fm', path=''):
        
        self.train_missing_modality_items = {}
        self.test_missing_modality_items = {}

        np.random.seed(seed)
        n_modality = len(fea)
        n_item = fea[0].shape[0]
        # random missing modality index
        protected_indices = np.random.randint(n_modality, size=n_item)

        # 1. first sample item from test set
        test_candidate_data = list(set(self.testItem))
        # print(len(test_candidate_data))
        test_num_missing_entries = int(len(test_candidate_data) * 0.5)
        np.random.shuffle(test_candidate_data)
        selected_for_test_missing = test_candidate_data[:test_num_missing_entries]
        # "selected_for_test_missing" these sampled items are randomly missing modality feature at inference step.
        self.test_missing_modality_items['items'] = selected_for_test_missing
        self.test_missing_modality_items['indicator'] = protected_indices[selected_for_test_missing]

        # 2. second sample item from train set via sample rate

        train_candidate_data = list(set(self.trainItem))
        # print(len(train_candidate_data))
        train_num_missing_entries = int(len(train_candidate_data) * rate)
        np.random.shuffle(train_candidate_data)
        selected_for_train_missing = train_candidate_data[:train_num_missing_entries]
        # "selected_for_train_missing" these sampled items are randomly missing modality feature at train step.
        self.train_missing_modality_items['items'] = selected_for_train_missing
        self.train_missing_modality_items['indicator'] = protected_indices[selected_for_train_missing]

        print('sample items with missing modality successfuly, train dataset include {0} items, test dataset include {1} items'.format(len(selected_for_train_missing), len(selected_for_test_missing)))


    def load_mutimedia_feature(self):
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

        self.set_miss_mutimedia_feature_items(
            fea,
            seed=0,
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

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
