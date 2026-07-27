import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mi_estimators import CLUBSample
import copy
# python milk_main.py --dataset baby --reg_coeff 1e-3 --penalty_coeff 100   --lr 1e-3  --exp_mode fm
# python milk_main.py --dataset baby --reg_coeff 1e-3 --penalty_coeff 300   --lr 1e-3  --exp_mode fm
# python milk_main.py --dataset baby --reg_coeff 1e-3 --penalty_coeff 700   --lr 1e-3  --exp_mode fm

class MGCN(torch.nn.Module):
    def __init__(self, edge_index, num_user, num_item, dim_feat, dim_latent):
        super(MGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.n_layers = 3

        self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
        
    def forward(self, features, user_id_preference):
        
        temp_features = self.MLP(features)
        all_emb = torch.cat((user_id_preference, temp_features),dim=0)
        all_emb = F.normalize(all_emb)
   
        embs = [all_emb]
        g_droped = self.edge_index    
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_user, self.num_item])
        
        return users, items



class MILK_model(torch.nn.Module):
    def __init__(self, env, dataset):
        super(MILK_model, self).__init__()
        self.env = env
        self.n_layers = 3
        self.A_split = False
        self.n_user = dataset.n_user
        self.m_item = dataset.m_item
        self.Graph = dataset.getSparseGraph()
        self.free_emb_dimension = self.env.args.free_emb_dimension

                
        self.train_missing_modality_items = dataset.train_missing_modality_items
        self.test_missing_modality_items = dataset.test_missing_modality_items

        self.audio_feat = None
        self.ori_audio_feat = None
        self.has_audio_modality = dataset.audio_feat is not None

        self.ori_image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        self.ori_image_feat = torch.nn.functional.normalize(self.ori_image_feat)
        self.v_gcn = MGCN(self.Graph, self.n_user, self.m_item, self.ori_image_feat.size(1), self.free_emb_dimension)


        self.ori_text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.ori_text_feat = torch.nn.functional.normalize(self.ori_text_feat)
        self.t_gcn = MGCN(self.Graph, self.n_user, self.m_item, self.ori_text_feat.size(1), self.free_emb_dimension)


        if self.has_audio_modality:
            self.ori_audio_feat = torch.tensor(dataset.audio_feat, dtype=torch.float32).to(self.env.device)
            self.ori_audio_feat = torch.nn.functional.normalize(self.ori_audio_feat)
            self.a_gcn = MGCN(self.Graph, self.n_user, self.m_item, self.ori_audio_feat.size(1), self.free_emb_dimension)


        self.user_emb = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.free_emb_dimension)
        self.item_emb = torch.nn.Embedding(
            num_embeddings=self.m_item, embedding_dim=self.free_emb_dimension)
        
        self.fusion_linear = nn.Sequential(
            nn.Linear(self.free_emb_dimension, self.free_emb_dimension, bias=False),
            nn.Dropout(),
            nn.Tanh()
        )

        self.final_item = None
        self.final_user = None
        self.activate = torch.nn.Sigmoid()

        # paramter initialization
        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        torch.nn.init.normal_(self.item_emb.weight, std=0.1)
        torch.nn.init.eye_(self.fusion_linear[0].weight)

        self.to(self.env.device)
        self.init_mi_estimator()
        self.init_missing_modality_set()

    def init_missing_modality_set(self):
            
            self.miss_test_image_feature = copy.deepcopy(self.ori_image_feat) 
            self.miss_test_text_feature = copy.deepcopy(self.ori_text_feat) 

            # init missing modality set for test dataset
            selected_missing_items = np.array(self.test_missing_modality_items['items']) 
            selected_missing_modality_indicator = np.array(self.test_missing_modality_items['indicator'])  

            # set image feature where some items are missing features
            image_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 0]
            self.miss_test_image_feature[image_missing_indicator] = 0

            # set text feature where some items are missing features
            text_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 1]
            self.miss_test_text_feature[text_missing_indicator] = 0

            if self.has_audio_modality:
                self.miss_test_audio_feature = copy.deepcopy(self.ori_audio_feat) 
                audio_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 2]
                self.miss_test_audio_feature[audio_missing_indicator] = 0


            # print(self.miss_test_image_feature[selected_missing_items[-1]], self.miss_test_text_feature[selected_missing_items[-1]])
            # exit()
            
            self.miss_train_image_feature = copy.deepcopy(self.ori_image_feat) 
            self.miss_train_text_feature = copy.deepcopy(self.ori_text_feat) 
            # init missing modality set for train dataset
            selected_missing_items = np.array(self.train_missing_modality_items['items'])
            selected_missing_modality_indicator = np.array(self.train_missing_modality_items['indicator'])
            
            image_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 0]
            self.miss_train_image_feature[image_missing_indicator] = 0

            text_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 1]
            self.miss_train_text_feature[text_missing_indicator] = 0

            if self.has_audio_modality:
                self.miss_train_audio_feature = copy.deepcopy(self.ori_audio_feat) 
                audio_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 2]
                self.miss_train_audio_feature[audio_missing_indicator] = 0

            # print(self.miss_train_image_feature[selected_missing_items[-1]], self.miss_train_text_feature[selected_missing_items[-1]])
            # exit()



    def set_missing_modality_via_env(self):
        # print(self.training)

        use_missing_train = self.env.args.exp_mode in {'mm', 'mf'}
        use_missing_test = self.env.args.exp_mode in {'mm', 'fm'}

        if self.training:
            if use_missing_train:
                self.image_feat = self.miss_train_image_feature
                self.text_feat = self.miss_train_text_feature
                if self.has_audio_modality:
                    self.audio_feat = self.miss_train_audio_feature
                print('set missing modality successfully for train setp')
            else:
                self.image_feat = self.ori_image_feat
                self.text_feat = self.ori_text_feat
                if self.has_audio_modality:
                    self.audio_feat = self.ori_audio_feat
                print('set complete modality successfully for train step')
        else:
            if use_missing_test:
                self.image_feat = self.miss_test_image_feature
                self.text_feat = self.miss_test_text_feature
                if self.has_audio_modality:
                    self.audio_feat = self.miss_test_audio_feature
                print('set missing modality successfully for test setp')
            else:
                self.image_feat = self.ori_image_feat
                self.text_feat = self.ori_text_feat
                if self.has_audio_modality:
                    self.audio_feat = self.ori_audio_feat
                print('set complete modality successfully for test step')


    def init_mi_estimator(self) :
        self.item_image_estimator = CLUBSample(self.ori_image_feat.size(1), self.free_emb_dimension, 64).to(self.env.device)
        self.item_text_estimator = CLUBSample(self.ori_text_feat.size(1), self.free_emb_dimension, 64).to(self.env.device)
        if self.has_audio_modality:
            self.item_audio_estimator = CLUBSample(self.ori_audio_feat.size(1), self.free_emb_dimension, 64).to(self.env.device)

        params = list(self.item_image_estimator.parameters()) + list(self.item_text_estimator.parameters())
        if self.has_audio_modality:
            params += list(self.item_audio_estimator.parameters())

        self.optimizer_club = torch.optim.Adam(params, lr = 1e-4)
        self.scheduler_club = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_club, 'min')

    def pre_epoch_processing(self) :

        item_image_g = self.image_feat
        item_text_g = self.text_feat
        item_audio_g = self.audio_feat

        with torch.no_grad():
            item_image_s = self.v_gcn.MLP(self.image_feat)
            item_text_s = self.t_gcn.MLP(self.text_feat)
            if self.has_audio_modality:
                item_audio_s = self.a_gcn.MLP(self.audio_feat)


        total_loss_mi = 0.0
        for _ in range(10) :
            self.item_image_estimator.train(); self.item_text_estimator.train()

            item_rand_idx = torch.randperm(self.m_item)[:2048]

            loss_mi = 0.0
        
            loss_mi += self.item_image_estimator.learning_loss(item_image_g[item_rand_idx], item_image_s[item_rand_idx])
            loss_mi += self.item_text_estimator.learning_loss(item_text_g[item_rand_idx], item_text_s[item_rand_idx])
            if self.has_audio_modality:
                loss_mi += self.item_audio_estimator.learning_loss(item_audio_g[item_rand_idx], item_audio_s[item_rand_idx])

            self.optimizer_club.zero_grad()
            loss_mi.backward(retain_graph = True)
            self.optimizer_club.step()
            total_loss_mi += loss_mi.detach().item()

        self.scheduler_club.step(total_loss_mi)
        self.item_image_estimator.eval(); self.item_text_estimator.eval()
        # print(total_loss_mi)


    def forward(self, random=False):
        """
        propagate methods for lightGCN
        """
        # self.set_missing_modality_via_env()

        user_id_emb = self.user_emb.weight
        v_user_emb, v_item_emb = self.v_gcn(self.image_feat, user_id_emb)
        t_user_emb, t_item_emb = self.t_gcn(self.text_feat, user_id_emb)


        if self.has_audio_modality:
            a_user_emb, a_item_emb = self.a_gcn(self.audio_feat, user_id_emb)

            user_emb = user_id_emb + (v_user_emb + t_user_emb + a_user_emb) / 3 
            item_emb = self.fusion_linear((v_item_emb.detach() + t_item_emb.detach() + a_item_emb.detach()) / 3)

        else:
            user_emb = user_id_emb + (v_user_emb + t_user_emb) / 2 
            item_emb = self.fusion_linear((v_item_emb.detach() + t_item_emb.detach()) / 2)

 
        # print(self.fusion_linear.weight)

        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0
        self.final_user = user_emb
        self.final_item = item_emb

        return user_emb, item_emb

    def get_env_emb(self, mix_ration, env):
        """
        propagate methods for lightGCN
        """
        # self.set_missing_modality_via_env()

        user_id_emb = self.user_emb.weight

        v_user_emb, v_item_emb = self.v_gcn(self.image_feat, user_id_emb)

        t_user_emb, t_item_emb = self.t_gcn(self.text_feat, user_id_emb)

        v_mm_emb = mix_ration[env][0] *  v_item_emb 
        t_mm_emb = mix_ration[env][1] *  t_item_emb 

        if self.has_audio_modality:
            a_user_emb, a_item_emb = self.a_gcn(self.audio_feat, user_id_emb)
            a_mm_emb = mix_ration[env][2] *  a_item_emb 
            
            item_emb = torch.cat([v_mm_emb, t_mm_emb, a_mm_emb], dim=1)
            user_emb = torch.cat([user_id_emb + v_user_emb, user_id_emb + t_user_emb, user_id_emb + a_user_emb], dim=1)
        else:
            item_emb = torch.cat([v_mm_emb, t_mm_emb], dim=1)
            user_emb = torch.cat([user_id_emb + v_user_emb, user_id_emb + t_user_emb], dim=1)


        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0

        return user_emb, item_emb



    def invariant_learning_emb(self, batch_users, batch_pos_items, batch_neg_items):

        max_mutual_loss = 0.
        user_id_emb = self.user_emb.weight

        v_user_emb, v_item_emb = self.v_gcn(self.image_feat, user_id_emb)
        t_user_emb, t_item_emb = self.t_gcn(self.text_feat, user_id_emb)

        v_item_detach_emb = v_item_emb.detach()
        t_item_detach_emb = t_item_emb.detach()
        

        user_emb = user_id_emb + (v_user_emb + t_user_emb) / 2

        if self.has_audio_modality:
            
            a_user_emb, a_item_emb = self.a_gcn(self.audio_feat, user_id_emb)

            user_emb = user_id_emb + (v_user_emb + t_user_emb + a_user_emb) / 3 

            a_item_detach_emb = a_item_emb.detach()

        

        if self.has_audio_modality:
            modality_mask = [[1, 1, 1], [1, 0, 0],[1, 1, 0],[1, 0, 1], [0, 1, 0],[0, 1, 1], [0, 0, 1]]
            mask_item_embs = []
            for mask_indication in modality_mask:

                mask_fusion_emb = (mask_indication[0] * v_item_detach_emb + mask_indication[1] * t_item_detach_emb + mask_indication[2] * a_item_detach_emb) / 3

                mask_item_emb = self.fusion_linear(mask_fusion_emb)
                mask_item_embs.append(mask_item_emb[batch_pos_items])

            mask_item_embs = torch.stack(mask_item_embs, dim=0)
            # print(mask_item_embs.shape)
        else:
            modality_mask = [[1,1], [1, 0], [0, 1]]

            mask_item_embs = []
            for mask_indication in modality_mask:

                mask_fusion_emb = (mask_indication[0] * v_item_detach_emb + mask_indication[1] * t_item_detach_emb) / 2

                mask_item_emb = self.fusion_linear(mask_fusion_emb)
                mask_item_embs.append(mask_item_emb[batch_pos_items])

            mask_item_embs = torch.stack(mask_item_embs, dim=0)
            


        # 互信息的最大化项
        for i in range(len(modality_mask)):
            for j in range(len(modality_mask)):
                if i != j:
                    max_mutual_loss += self.InfoNCE(mask_item_embs[i], mask_item_embs[j])

        max_mutual_loss += max_mutual_loss / 6

        # 互信息的最小化项

        item_emb = self.fusion_linear((v_item_emb.detach() + t_item_emb.detach()) / 2)

        if self.has_audio_modality:
            item_emb = self.fusion_linear((v_item_emb.detach() + t_item_emb.detach() + a_item_emb.detach()) / 3)

        item_image_g = self.image_feat
        item_text_g = self.text_feat
        loss_club = 0.0
        loss_club += self.item_image_estimator(item_image_g[batch_pos_items], item_emb[batch_pos_items])
        loss_club += self.item_text_estimator(item_text_g[batch_pos_items], item_emb[batch_pos_items])

        min_mutual_loss = (loss_club/2)

        if self.has_audio_modality:
            item_audio_g = self.audio_feat
            loss_club += self.item_audio_estimator(item_audio_g[batch_pos_items], item_emb[batch_pos_items])
            min_mutual_loss = (loss_club/3)

       

        # item_emb_cl_loss = self.InfoNCE(item_emb[batch_pos_items], v_item_emb[batch_pos_items].detach()) + self.InfoNCE(item_emb[batch_pos_items], t_item_emb[batch_pos_items].detach())

        # max_mutual_loss = (item_emb_cl_loss/2)

        all_user_emb, all_item_emb =  self.forward()

        user_emb  = all_user_emb[batch_users]
        pos_items = all_item_emb[batch_pos_items]
        neg_items = all_item_emb[batch_neg_items]

        pos_scores = torch.sum(torch.mul(user_emb, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_items), dim=1)

        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        return bpr_loss, max_mutual_loss, min_mutual_loss

    def basic_recommendation_loss(self, batch_users, batch_pos_items, batch_neg_items):
        all_user_emb, all_item_emb = self.forward()

        user_emb = all_user_emb[batch_users]
        pos_items = all_item_emb[batch_pos_items]
        neg_items = all_item_emb[batch_neg_items]

        pos_scores = torch.sum(torch.mul(user_emb, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_items), dim=1)

        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    def InfoNCE(self, view1, view2, temperature = 0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)


    def calculate_reg_loss(self, batch_users, batch_pos_items, batch_neg_items):

        all_user_emb, all_item_emb =  self.forward()
        # all_user_emb = self.user_emb.weight
        # all_item_emb = self.item_emb.weight

        reg_embedding_loss = (1 / 2) * (all_user_emb[batch_users].norm(2).pow(2) + all_item_emb[batch_pos_items].norm(2).pow(2) + all_item_emb[batch_neg_items].norm(2).pow(2)) / float(len(batch_users))

        return reg_embedding_loss
