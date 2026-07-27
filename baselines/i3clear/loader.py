import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
from time import time

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

        trainUniqueUsers, trainItem, trainUser =[], [], []
        valUniqueUsers, valItem, valUser =[], [], []
        testUniqueUsers, testItem, testUser =[], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        self.train_data = defaultdict(list)
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]=='':
                        continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_data[uid].extend(items)
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        
      


        setTrainItem = set(trainItem)
        self.cold_item_index = set()
        self.val_data = defaultdict(list)
        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
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

        self.test_data = defaultdict(list)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
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

        self.UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)), shape=(self.n_user, self.m_item))

        all_item_index = np.array(list(range(self.m_item)))
        # self.cold_item_index = np.load(os.path.join(self.env.DATA_PATH, 'cold_item_index.npy'))
        # self.warm_item_index = np.setdiff1d(all_item_index, self.cold_item_index, assume_unique=False)
        # self.warm_missing_item_index = np.load(os.path.join(self.env.DATA_PATH, 'warm_missing_item_index.npy'))
        # self.cold_missing_item_index = np.load(os.path.join(self.env.DATA_PATH, 'cold_missing_item_index.npy'))

        self.image_feat, self.text_feat, self.audio_feat = self.load_mutimedia_feature()
        self.feature = np.concatenate([self.image_feat, self.text_feat], axis=1)
        if self.env.args.dataset == 'tiktok':
            self.feature = np.concatenate([self.feature, self.audio_feat], axis=1)
            

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

    def load_mutimedia_feature(self):
        if self.env.args.dataset != 'tiktok':
            
            if self.env.args.exp_mode=='ff':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
                text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')
            elif self.env.args.exp_mode=='fm':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_test.npy')
                text_file = os.path.join(self.env.DATA_PATH, 'text_feat_missing_test.npy')
            elif self.env.args.exp_mode=='mm':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_moD3.npy')
                text_file = os.path.join(self.env.DATA_PATH,  'text_feat_missing_moD3.npy')
            elif self.env.args.exp_mode=='mm1':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_moD0.1.npy')
                text_file = os.path.join(self.env.DATA_PATH,  'text_feat_missing_moD0.1.npy')
            elif self.env.args.exp_mode=='mm3':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_moD0.3.npy')
                text_file = os.path.join(self.env.DATA_PATH,  'text_feat_missing_moD0.3.npy')
            elif self.env.args.exp_mode=='mm5':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_moD0.5.npy')
                text_file = os.path.join(self.env.DATA_PATH,  'text_feat_missing_moD0.5.npy')
            elif self.env.args.exp_mode=='mm7':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_moD0.7.npy')
                text_file = os.path.join(self.env.DATA_PATH,  'text_feat_missing_moD0.7.npy')
            elif self.env.args.exp_mode=='mm9':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_moD0.9.npy')
                text_file = os.path.join(self.env.DATA_PATH,  'text_feat_missing_moD0.9.npy')
            elif self.env.args.exp_mode=='wot':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
                text_file = os.path.join(self.env.DATA_PATH,  'text_feat_missing_moD0.5.npy')
            elif self.env.args.exp_mode=='wom':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_moD0.5.npy')
                text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')
        
        
        else:
            if self.env.args.exp_mode=='ff':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
                text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')
                audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat.npy')
            elif self.env.args.exp_mode=='fm':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_test.npy')
                text_file = os.path.join(self.env.DATA_PATH, 'text_feat_missing_test.npy')
                audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat_missing_test.npy')
            elif self.env.args.exp_mode=='mf':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_train.npy')
                text_file = os.path.join(self.env.DATA_PATH, 'text_feat_missing_train.npy')
                audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat_missing_train.npy')
            elif self.env.args.exp_mode=='mm':
                image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_all.npy')
                text_file = os.path.join(self.env.DATA_PATH, 'text_feat_missing_all.npy')
                audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat_missing_all.npy')


        # low_dim_feature_file = os.path.join(self.env.DATA_PATH, 'low_dim_feature.npy')
        image_feat = np.load(image_file)
        text_feat = np.load(text_file)
        if self.env.args.dataset == 'tiktok':
            audio_feat = np.load(audio_file)
        # low_dim_feature = np.load(low_dim_feature_file)
        if self.env.args.dataset == 'tiktok':
            return image_feat, text_feat, audio_feat
        else:
            return image_feat, text_feat, None


    def getUserAllItems(self):
        """
        得到训练集和验证集中用户交互过的item
        用于负采样
        """
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

