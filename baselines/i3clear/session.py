import os
import time
import random
import torch
from torch import autograd
from tqdm import tqdm
from collections import defaultdict

import numpy as np

import criterion
import tool
import dataset_loader
import evaluation
# from metric import evaluation


class MILK_session(object):

    def __init__(self, env, model, loader):
        self.env = env
        self.model = model
        self.dataset = loader

        # preference_aware_params = list(self.model.v_gcn.parameters()) + list(self.model.t_gcn.parameters()) + list(self.model.id_embedding.parameters())

        # self.representation_optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr}])

        self.representation_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.env.args.lr)

        self.bpr = criterion.BPR()
        self.align = criterion.MSE()
        self.early_stop = 0
        self.best_epoch = 0
        self.total_epoch = 0
        self.best_ndcg = defaultdict(float)
        self.best_hr = defaultdict(float)
        self.best_recall = defaultdict(float)
        self.test_ndcg = defaultdict(float)
        self.test_hr = defaultdict(float)
        self.test_recall = defaultdict(float)
        
    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1
        S = dataset_loader.PairSample(self.dataset)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()
        users = users.to(self.env.device)
        posItems = posItems.to(self.env.device)
        negItems = negItems.to(self.env.device)
        users, posItems, negItems = tool.shuffle(users, posItems, negItems)
        total_batch = len(users) // self.env.args.batch_size + 1
        all_loss, all_main_bpr_loss, all_maximize_loss = 0., 0., 0.
        all_modality_bpr_loss, all_grad_loss, all_penalty_loss = 0., 0., 0.
        self.model.set_missing_modality_via_env()
        if not self.env.args.disable_ib:
            self.model.pre_epoch_processing()
        for user, pos_item, neg_item, in dataset_loader.minibatch(users,
                                                             posItems,
                                                             negItems,
                                                             batch_size=self.env.args.batch_size):

            n_modalities = 3 if getattr(self.model, "has_audio_modality", False) else 2
            mix_ration = [
                [1 if i == j else 0 for i in range(n_modalities)]
                for j in range(n_modalities)
            ]
           
            env_penalty = []
            env_reg = []
            for i in range(len(mix_ration)):
                env_user_emb, env_item_emb = self.model.get_env_emb(mix_ration, i)
                env_bpr_loss, env_reg_loss = self.bpr(env_user_emb, env_item_emb, user, pos_item, neg_item, model_ego = self.model)                
                env_penalty.append(env_bpr_loss)
                env_reg.append(env_reg_loss)
            penalty_loss = torch.stack(env_penalty).var()
            # rbpr_loss = torch.stack(env_penalty).mean()

            # print((env_penalty[0].sum()- env_penalty[1].sum()))

            penalty_loss = self.env.args.penalty_coeff * penalty_loss
            if self.env.args.disable_irm:
                penalty_loss = torch.zeros_like(penalty_loss)
            reg_loss = self.env.args.reg_coeff * env_reg[0]
            modality_bpr_loss = sum(env_penalty) 
            
            if self.env.args.disable_ib:
                main_bpr_loss = self.model.basic_recommendation_loss(user, pos_item, neg_item)
                maximize_mutual_info = torch.zeros_like(main_bpr_loss)
                minimize_mutual_info = torch.zeros_like(main_bpr_loss)
            else:
                main_bpr_loss, maximize_mutual_info, minimize_mutual_info = self.model.invariant_learning_emb(user, pos_item, neg_item)
            # print(maximize_mutual_info)
            mutual_info = self.env.args.max_info_coeff * (maximize_mutual_info) + self.env.args.min_info_coeff * minimize_mutual_info
            # mutual_info = self.env.args.max_info_coeff * (maximize_mutual_info)
            # print(mutual_info)
            # # loss = main_bpr_loss + maximize_mutual_info + modality_bpr_loss + reg_loss + penalty_loss

            loss = main_bpr_loss + mutual_info + modality_bpr_loss + reg_loss + penalty_loss

            # print(self.model.id_embedding.weight.mean())

            self.representation_optimizer.zero_grad()
            loss.backward()
            self.representation_optimizer.step()

            all_loss += loss
            all_main_bpr_loss += main_bpr_loss
            all_maximize_loss += 0.
            all_modality_bpr_loss += modality_bpr_loss
            all_grad_loss += reg_loss
            all_penalty_loss += penalty_loss
        return all_loss / total_batch, all_main_bpr_loss / total_batch, all_maximize_loss / total_batch, all_modality_bpr_loss/total_batch, all_grad_loss/total_batch, all_penalty_loss/total_batch, time.time() - t
 
    def train(self, epochs):
        # self.model.init_mi_estimator()
        for epoch in range(self.env.args.ckpt_start_epoch, epochs):
            self.model.train()
            loss, main_bpr_loss, maximize_loss, modality_bpr_loss, reg_loss, penalty_loss, train_time = self.train_epoch()
            # self.model.show_scores()
            print('-' * 30)
            print(
                f'TRAIN:epoch = {epoch}/{epochs} loss_s1 = {loss:.5f}, main_bpr_loss = {main_bpr_loss:.5f}, modality_bpr_loss = {modality_bpr_loss:.5f}, maximize_loss={maximize_loss:.5f}, penalty_loss = {penalty_loss:.5f}, reg_loss = {reg_loss:.5f},  train_time = {train_time:.2f}')

            # self.model.eval()
            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += self.env.args.eva_interval
                hr, recall, ndcg, val_time = self.test(mode='val', top_list=eval(self.env.args.topk))
                if self.env.args.tensorboard:
                    for key in hr.keys():
                        self.env.w.add_scalar(
                            f'Val/hr@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/recall@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/ndcg@{key}', ndcg[key], self.total_epoch)
                key = list(hr.keys())[0]
                print(
                    f'epoch = {epoch} hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

                if ndcg[list(hr.keys())[0]] > self.best_ndcg[list(hr.keys())[0]]:
                    previous_best_epoch = self.best_epoch
                    thr, trecall, tndcg, test_time = self.test(mode='test', top_list=eval(self.env.args.topk))
                    self.early_stop = 0
                    for key in thr.keys():
                        tool.cprint(
                            f'epoch = {epoch} hr@{key} = {thr[key]:.5f}, recall@{key} = {trecall[key]:.5f}, ndcg@{key} = {tndcg[key]:.5f}, test_time = {test_time:.2f}')
                    tool.cprint('----------------------')

                    for key in hr.keys():
                        self.best_hr[key] = hr[key]
                        self.best_recall[key] = recall[key]
                        self.best_ndcg[key] = ndcg[key]
                    for key in thr.keys():
                        self.test_hr[key] = thr[key]
                        self.test_recall[key] = trecall[key]
                        self.test_ndcg[key] = tndcg[key]
                    self.best_epoch = epoch
                    if self.env.args.save:
                        self.save_model(epoch, previous_best_epoch=previous_best_epoch)
                        print('save ckpt')
                    if self.env.args.log:
                        self.env.val_logger.info(f'EPOCH[{epoch}/{epochs}]')
                        for key in hr.keys():
                            self.env.val_logger.info(
                                f'hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

            # if self.env.args.log:
            #     self.env.train_logger.info(
            #         f'EPOCH[{epoch}/{epochs}], loss = {loss:.5f}, bpr_loss = {bpr_loss:.5f}, reg_loss = {reg_loss:.5f}')

            # if self.env.args.tensorboard:
            #     self.env.w.add_scalar(f'Train/loss', loss, self.total_epoch)
            #     self.env.w.add_scalar(
            #         f'Train/bpr_loss', bpr_loss, self.total_epoch)
            #     self.env.w.add_scalar(
            #         f'Train/reg_loss', reg_loss, self.total_epoch)

            if self.early_stop > self.env.args.early_stop // 1:
                break


    def test(self, mode='val', top_list=[50]):
        self.model.eval()
        self.model.set_missing_modality_via_env()
        t = time.time()
        # user_emb = self.model.user_emb.weight
        # image_feat = self.model.image_feat
        # text_feat = self.model.text_feat
        # item_emb = (self.model.image_linear(image_feat) + self.model.text_linear(text_feat))/2

        user_emb, item_emb = self.model()

        user_emb = user_emb.cpu().detach().numpy()
        item_emb = item_emb.cpu().detach().numpy()
        if mode == 'val':
            hr, recall, ndcg = evaluation.num_faiss_evaluate(self.dataset.val_data,
                                                        list(
                                                            self.dataset.val_data.keys()),
                                                        list(
                                                            self.dataset.cold_item_index),
                                                        self.dataset.train_data,
                                                        top_list, user_emb, item_emb)
        else:
            hr, recall, ndcg = evaluation.num_faiss_evaluate(self.dataset.test_data,
                                                             list(
                                                                     self.dataset.test_data.keys()),
                                                            list(
                                                                self.dataset.cold_item_index),
                                                             self.dataset.train_data,
                                                             top_list, user_emb, item_emb)

        return hr, recall, ndcg, time.time() - t

    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch, previous_best_epoch=None):
        model_state_file = os.path.join(
            self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        if previous_best_epoch is not None and current_epoch != previous_best_epoch:
            old_model_state_file = os.path.join(
                self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{previous_best_epoch}.pth')
            if os.path.exists(old_model_state_file):
                os.system('rm {}'.format(old_model_state_file))
