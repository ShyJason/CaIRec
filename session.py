import os
import time
import copy
import torch
from collections import defaultdict

import numpy as np

import tool
import dataset_loader
import evaluation
# from metric import evaluation

STAGE1_IMPUTER_STAGES = (
    'imputer_param',
    'imputer_backprop',
)


class MILK_session(object):

    def __init__(self, env, model, loader):
        self.env = env
        self.model = model
        self.dataset = loader

        self.model.configure_training_stage(
            self.env.args.train_stage,
            freeze_imputer=self.env.args.freeze_imputer,
            freeze_recommender=self.env.args.freeze_recommender,
        )
        self.representation_optimizer = self._build_representation_optimizer()

        self.early_stop = 0
        self.best_epoch = 0
        self.total_epoch = 0
        self.best_ndcg = defaultdict(float)
        self.best_hr = defaultdict(float)
        self.best_recall = defaultdict(float)
        self.test_ndcg = defaultdict(float)
        self.test_hr = defaultdict(float)
        self.test_recall = defaultdict(float)
        self.best_recommendation_metric = float('-inf')
        self.stage_name = self.env.args.train_stage
        self.last_train_metrics = {}
        self.best_model_state = None
        self.last_test_modality_subsets = {}

    def _resolve_recommendation_selection(self, hr, recall, ndcg):
        topk = self.env.args.recommendation_selection_topk
        if topk not in hr:
            topk = 20 if 20 in hr else list(hr.keys())[0]

        metric_name = self.env.args.recommendation_selection_metric
        metric_map = {
            'hr': hr,
            'recall': recall,
            'ndcg': ndcg,
        }
        metric_value = float(metric_map[metric_name][topk])
        return metric_name, topk, metric_value

    def _resolve_stage_learning_rates(self):
        lr_rec = self.env.args.lr_rec if self.env.args.lr_rec is not None else self.env.args.lr
        if self.env.args.lr_imp is not None:
            lr_imp = self.env.args.lr_imp
        else:
            lr_imp = lr_rec

        return lr_rec, lr_imp

    def _build_representation_optimizer(self):
        lr_rec, lr_imp = self._resolve_stage_learning_rates()
        param_groups = []

        imputer_params = self.model.get_imputer_parameters()
        if imputer_params:
            param_groups.append({'params': imputer_params, 'lr': lr_imp})

        recommender_params = self.model.get_recommender_parameters()
        if recommender_params:
            param_groups.append({'params': recommender_params, 'lr': lr_rec})

        if not param_groups:
            return None

        return torch.optim.Adam(param_groups, lr=lr_rec)

    def switch_training_stage(self, train_stage, freeze_imputer=-1, freeze_recommender=-1):
        self.env.args.train_stage = train_stage
        self.env.args.freeze_imputer = freeze_imputer
        self.env.args.freeze_recommender = freeze_recommender
        self.model.configure_training_stage(
            self.env.args.train_stage,
            freeze_imputer=self.env.args.freeze_imputer,
            freeze_recommender=self.env.args.freeze_recommender,
        )
        self.representation_optimizer = self._build_representation_optimizer()
        self.stage_name = self.env.args.train_stage

    def _recommendation_best_state(self):
        return {
            'best_epoch': self.best_epoch,
            'best_recommendation_metric': self.best_recommendation_metric,
            'best_hr': copy.deepcopy(self.best_hr),
            'best_recall': copy.deepcopy(self.best_recall),
            'best_ndcg': copy.deepcopy(self.best_ndcg),
            'test_hr': copy.deepcopy(self.test_hr),
            'test_recall': copy.deepcopy(self.test_recall),
            'test_ndcg': copy.deepcopy(self.test_ndcg),
            'best_model_state': copy.deepcopy(self.best_model_state),
            'early_stop': self.early_stop,
        }

    def _restore_recommendation_best_state(self, state):
        self.best_epoch = state['best_epoch']
        self.best_recommendation_metric = state['best_recommendation_metric']
        self.best_hr = state['best_hr']
        self.best_recall = state['best_recall']
        self.best_ndcg = state['best_ndcg']
        self.test_hr = state['test_hr']
        self.test_recall = state['test_recall']
        self.test_ndcg = state['test_ndcg']
        self.best_model_state = state['best_model_state']
        self.early_stop = state['early_stop']

    def _zero_scalar(self):
        return torch.zeros((), device=self.env.device)

    def _snapshot_best_model_state(self):
        self.best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _restore_model_state(self, state_dict):
        restored = {
            key: value.to(self.env.device)
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(restored, strict=True)

    def _finalize_strict_test_metrics(self):
        if self.best_model_state is None:
            return

        current_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        was_training = self.model.training
        try:
            self._restore_model_state(self.best_model_state)
            thr, trecall, tndcg, test_time = self.test(mode='test', top_list=self.env.args.topk)
            for key in thr.keys():
                tool.cprint(
                    f'final strict test hr@{key} = {thr[key]:.5f}, recall@{key} = {trecall[key]:.5f}, ndcg@{key} = {tndcg[key]:.5f}, test_time = {test_time:.2f}'
                )
            for key in thr.keys():
                self.test_hr[key] = thr[key]
                self.test_recall[key] = trecall[key]
                self.test_ndcg[key] = tndcg[key]
            self.log_test_modality_subsets(prefix='final strict test')
        finally:
            self._restore_model_state(current_state)
            if was_training:
                self.model.train()

    def _compute_stage_promrl_loss(self, promrl_losses):
        stage = self.env.args.train_stage
        zero = self._zero_scalar()
        if promrl_losses is None:
            return zero, zero, zero, zero, zero

        promrl_intra_loss = promrl_losses['loss_intra']
        promrl_inter_loss = promrl_losses['loss_inter']
        promrl_itm_loss = promrl_losses['loss_itm']
        promrl_rec_loss = promrl_losses['rec_loss']

        if stage == 'imputer_param':
            stage_promrl_loss = self.env.args.alpha_rec * promrl_rec_loss
            promrl_intra_loss = zero
            promrl_inter_loss = zero
            promrl_itm_loss = zero
        elif stage == 'imputer_backprop':
            stage_promrl_loss = (
                self.env.args.alpha_intra * promrl_intra_loss
                + self.env.args.alpha_inter * promrl_inter_loss
                + self.env.args.alpha_itm * promrl_itm_loss
                + self.env.args.alpha_rec * promrl_rec_loss
            )
        else:
            stage_promrl_loss = zero
            promrl_intra_loss = zero
            promrl_inter_loss = zero
            promrl_itm_loss = zero
            promrl_rec_loss = zero

        return (
            stage_promrl_loss,
            promrl_intra_loss,
            promrl_inter_loss,
            promrl_itm_loss,
            promrl_rec_loss,
        )

    def _should_compute_promrl_losses(self, effective_stage):
        if effective_stage not in (
            'imputer_param',
            'imputer_backprop',
        ):
            return False

        weights = (
            self.env.args.alpha_intra,
            self.env.args.alpha_inter,
            self.env.args.alpha_itm,
            self.env.args.alpha_rec,
        )

        return any(float(weight) != 0.0 for weight in weights)
        
    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1
        current_epoch = self.total_epoch - 1
        self.model.current_epoch = current_epoch
        stage = self.env.args.train_stage
        imputer_only_stage = stage in STAGE1_IMPUTER_STAGES
        if imputer_only_stage:
            item_ids = torch.as_tensor(
                dataset_loader.ItemSample(self.dataset),
                dtype=torch.long,
                device=self.env.device,
            )
            item_ids = tool.shuffle(item_ids)
            total_batch = len(item_ids) // self.env.args.batch_size + 1
            users = posItems = negItems = None
        else:
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
        all_promrl_intra_loss, all_promrl_inter_loss = 0., 0.
        all_promrl_itm_loss, all_promrl_rec_loss = 0., 0.
        all_align_loss = 0.
        self.model.set_missing_modality_via_env()
        if imputer_only_stage:
            batch_iterator = (
                (None, batch_item_ids, None)
                for batch_item_ids in dataset_loader.minibatch(
                    item_ids,
                    batch_size=self.env.args.batch_size,
                )
            )
        else:
            batch_iterator = dataset_loader.minibatch(
                users,
                posItems,
                negItems,
                batch_size=self.env.args.batch_size,
            )

        for batch_index, (user, pos_item, neg_item) in enumerate(batch_iterator):
            zero = self._zero_scalar()
            promrl_losses = None
            effective_stage = self.model._canonical_stage(stage)
            if self._should_compute_promrl_losses(effective_stage):
                if imputer_only_stage:
                    promrl_item_ids = pos_item
                else:
                    promrl_item_ids = torch.unique(torch.cat([pos_item, neg_item], dim=0))
                promrl_losses = self.model.compute_promrl_losses(promrl_item_ids, stage=effective_stage)

            (
                stage_promrl_loss,
                promrl_intra_loss,
                promrl_inter_loss,
                promrl_itm_loss,
                promrl_rec_loss,
            ) = \
                self._compute_stage_promrl_loss(promrl_losses)

            self.model.clear_gcn_cache()

            main_bpr_loss = zero
            modality_bpr_loss = zero
            reg_loss = zero
            penalty_loss = zero
            mutual_info = zero
            align_loss = zero

            if effective_stage == 'recommender':
                allow_recommender_modal_grad = bool(getattr(self.env.args, 'recommender_allow_modal_grad', 0))
                all_user_emb = None
                all_item_emb = None
                if allow_recommender_modal_grad:
                    main_bpr_loss, base_reg_loss, all_user_emb, all_item_emb = self.model.basic_recommendation_loss(
                        user,
                        pos_item,
                        neg_item,
                        allow_modal_grad=True,
                        return_embeddings=True,
                    )
                else:
                    main_bpr_loss, base_reg_loss = self.model.basic_recommendation_loss(
                        user, pos_item, neg_item, allow_modal_grad=False
                    )
                if self.env.args.modality_bpr_coeff > 0:
                    modality_bpr_loss = (
                        self.env.args.modality_bpr_coeff
                        * self.model.modality_bpr_loss(user, pos_item, neg_item)
                    )
                if float(getattr(self.env.args, 'gamma_align', 0.0)) > 0.0:
                    align_items = torch.unique(torch.cat([pos_item, neg_item], dim=0))
                    align_loss = self.model.compute_adapter_alignment_loss(
                        align_items,
                        pseudo_ratio=float(getattr(self.env.args, 'adapter_align_pseudo_ratio', 1.0)),
                    )
                self.model.clear_gcn_cache()
                reg_loss = self.env.args.reg_coeff * base_reg_loss

            loss = (
                main_bpr_loss
                + mutual_info
                + modality_bpr_loss
                + reg_loss
                + penalty_loss
                + stage_promrl_loss
                + self.env.args.gamma_align * align_loss
            )

            # print(self.model.id_embedding.weight.mean())

            if self.representation_optimizer is not None:
                self.representation_optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                self.representation_optimizer.step()
            if self.model.has_pending_em_updates():
                self.model.apply_pending_em_updates()

            all_loss += loss
            all_main_bpr_loss += main_bpr_loss
            all_maximize_loss += 0.
            all_modality_bpr_loss += modality_bpr_loss
            all_grad_loss += reg_loss
            all_penalty_loss += penalty_loss
            all_promrl_intra_loss += promrl_intra_loss
            all_promrl_inter_loss += promrl_inter_loss
            all_promrl_itm_loss += promrl_itm_loss
            all_promrl_rec_loss += promrl_rec_loss
            all_align_loss += align_loss
        return (
            all_loss / total_batch,
            all_main_bpr_loss / total_batch,
            all_maximize_loss / total_batch,
            all_modality_bpr_loss / total_batch,
            all_grad_loss / total_batch,
            all_penalty_loss / total_batch,
            all_promrl_intra_loss / total_batch,
            all_promrl_inter_loss / total_batch,
            all_promrl_itm_loss / total_batch,
            all_promrl_rec_loss / total_batch,
            all_align_loss / total_batch,
            time.time() - t,
        )

    def train(self, epochs, finalize=True, start_epoch=None):
        strict_probe_test_interval = int(getattr(self.env.args, 'strict_probe_test_interval', 0) or 0)
        if bool(getattr(self.env.args, 'strict_record_test_each_epoch', 0)):
            strict_probe_test_interval = self.env.args.eva_interval
        strict_probe_test_interval = max(0, strict_probe_test_interval)
        record_strict_test_probe = (
            self.env.args.evaluation_protocol == 'strict'
            and strict_probe_test_interval > 0
        )
        if start_epoch is None:
            start_epoch = self.env.args.ckpt_start_epoch
        for epoch in range(start_epoch, epochs):
            self.model.train()
            (
                loss,
                main_bpr_loss,
                maximize_loss,
                modality_bpr_loss,
                reg_loss,
                penalty_loss,
                promrl_intra_loss,
                promrl_inter_loss,
                promrl_itm_loss,
                promrl_rec_loss,
                align_loss,
                train_time,
            ) = self.train_epoch()
            # self.model.show_scores()
            print('-' * 30)
            print(
                f'TRAIN:stage = {self.stage_name}, epoch = {epoch}/{epochs} loss_s1 = {loss:.5f}, main_bpr_loss = {main_bpr_loss:.5f}, modality_bpr_loss = {modality_bpr_loss:.5f}, maximize_loss={maximize_loss:.5f}, penalty_loss = {penalty_loss:.5f}, reg_loss = {reg_loss:.5f}, promrl_intra = {promrl_intra_loss:.5f}, promrl_inter = {promrl_inter_loss:.5f}, promrl_itm = {promrl_itm_loss:.5f}, promrl_rec = {promrl_rec_loss:.5f}, align_loss = {align_loss:.5f}, train_time = {train_time:.2f}')
            self.last_train_metrics = {
                'epoch': epoch,
                'loss_s1': float(loss),
                'main_bpr_loss': float(main_bpr_loss),
                'modality_bpr_loss': float(modality_bpr_loss),
                'maximize_loss': float(maximize_loss),
                'penalty_loss': float(penalty_loss),
                'reg_loss': float(reg_loss),
                'promrl_intra': float(promrl_intra_loss),
                'promrl_inter': float(promrl_inter_loss),
                'promrl_itm': float(promrl_itm_loss),
                'promrl_rec': float(promrl_rec_loss),
                'align_loss': float(align_loss),
                'train_time': float(train_time),
            }
            if self.env.args.tensorboard:
                self.env.w.add_scalar('Train/loss', float(loss), self.total_epoch)
                self.env.w.add_scalar('Train/main_bpr_loss', float(main_bpr_loss), self.total_epoch)
                self.env.w.add_scalar('Train/modality_bpr_loss', float(modality_bpr_loss), self.total_epoch)
                self.env.w.add_scalar('Train/penalty_loss', float(penalty_loss), self.total_epoch)
                self.env.w.add_scalar('Train/reg_loss', float(reg_loss), self.total_epoch)
                self.env.w.add_scalar('Train/promrl_intra', float(promrl_intra_loss), self.total_epoch)
                self.env.w.add_scalar('Train/promrl_inter', float(promrl_inter_loss), self.total_epoch)
                self.env.w.add_scalar('Train/promrl_itm', float(promrl_itm_loss), self.total_epoch)
                self.env.w.add_scalar('Train/promrl_rec', float(promrl_rec_loss), self.total_epoch)
                self.env.w.add_scalar('Train/align_loss', float(align_loss), self.total_epoch)
                for group_idx, group in enumerate(self.representation_optimizer.param_groups if self.representation_optimizer is not None else []):
                    self.env.w.add_scalar(f'Train/lr_group_{group_idx}', group['lr'], self.total_epoch)

            if self.env.args.train_stage in STAGE1_IMPUTER_STAGES:
                if epoch == epochs - 1 and self.env.args.save:
                    self.best_epoch = epoch
                    self.save_model(epoch)
                    print('save final Stage 1 checkpoint')
                continue

            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += self.env.args.eva_interval
                selection_mode = self.env.args.selection_mode
                hr, recall, ndcg, val_time = self.test(mode=selection_mode, top_list=self.env.args.topk)
                if self.env.args.tensorboard:
                    for key in hr.keys():
                        self.env.w.add_scalar(
                            f'Val/hr_{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/recall_{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/ndcg_{key}', ndcg[key], self.total_epoch)
                key = 20 if 20 in hr else list(hr.keys())[0]
                print(
                    f'epoch = {epoch} hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, {selection_mode}_time = {val_time:.2f}')
                metric_name, metric_topk, metric_value = self._resolve_recommendation_selection(hr, recall, ndcg)
                if metric_value > self.best_recommendation_metric:
                    self.early_stop = 0
                    self.best_recommendation_metric = metric_value
                    for key in hr.keys():
                        self.best_hr[key] = hr[key]
                        self.best_recall[key] = recall[key]
                        self.best_ndcg[key] = ndcg[key]
                    self.best_epoch = epoch
                    if self.env.args.evaluation_protocol == 'strict':
                        self._snapshot_best_model_state()
                    else:
                        if selection_mode == 'test':
                            thr, trecall, tndcg, test_time = hr, recall, ndcg, val_time
                        else:
                            thr, trecall, tndcg, test_time = self.test(mode='test', top_list=self.env.args.topk)
                        for key in thr.keys():
                            tool.cprint(
                                f'epoch = {epoch} hr@{key} = {thr[key]:.5f}, recall@{key} = {trecall[key]:.5f}, ndcg@{key} = {tndcg[key]:.5f}, test_time = {test_time:.2f}')
                        tool.cprint('----------------------')

                        if self.env.args.tensorboard:
                            for key in thr.keys():
                                self.env.w.add_scalar(f'Test/hr_{key}', thr[key], self.total_epoch)
                                self.env.w.add_scalar(f'Test/recall_{key}', trecall[key], self.total_epoch)
                                self.env.w.add_scalar(f'Test/ndcg_{key}', tndcg[key], self.total_epoch)

                        for key in thr.keys():
                            self.test_hr[key] = thr[key]
                            self.test_recall[key] = trecall[key]
                            self.test_ndcg[key] = tndcg[key]
                    if self.env.args.save:
                        self.save_model(epoch)
                        print(f'save ckpt (best {selection_mode} {metric_name}@{metric_topk})')
                    if self.env.args.log:
                        self.env.val_logger.info(f'EPOCH[{epoch}/{epochs}]')
                        for key in hr.keys():
                            self.env.val_logger.info(
                                f'hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, {selection_mode}_time = {val_time:.2f}')

            should_probe_test = (
                record_strict_test_probe
                and self.env.args.train_stage not in STAGE1_IMPUTER_STAGES
                and (epoch + 1) % strict_probe_test_interval == 0
            )
            if should_probe_test:
                test_hr, test_recall, test_ndcg, test_time = self.test(mode='test', top_list=self.env.args.topk)
                test_key = 20 if 20 in test_hr else list(test_hr.keys())[0]
                print(
                    f'strict_probe epoch = {epoch} hr@{test_key} = {test_hr[test_key]:.5f}, '
                    f'recall@{test_key} = {test_recall[test_key]:.5f}, ndcg@{test_key} = {test_ndcg[test_key]:.5f}, '
                    f'test_time = {test_time:.2f}'
                )
                if self.env.args.tensorboard:
                    for test_metric_key in test_hr.keys():
                        self.env.w.add_scalar(
                            f'StrictProbeTest/hr_{test_metric_key}', test_hr[test_metric_key], self.total_epoch
                        )
                        self.env.w.add_scalar(
                            f'StrictProbeTest/recall_{test_metric_key}', test_recall[test_metric_key], self.total_epoch
                        )
                        self.env.w.add_scalar(
                            f'StrictProbeTest/ndcg_{test_metric_key}', test_ndcg[test_metric_key], self.total_epoch
                        )

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

        if (
            finalize
            and self.env.args.evaluation_protocol == 'strict'
            and self.env.args.train_stage not in STAGE1_IMPUTER_STAGES
        ):
            self._finalize_strict_test_metrics()

    def test(self, mode='val', top_list=[50]):
        self.model.eval()
        self.model.set_missing_modality_via_env(eval_split=mode)
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
                                                        self.dataset.train_data,
                                                        top_list, user_emb, item_emb)
        else:
            hr, recall, ndcg = evaluation.num_faiss_evaluate(self.dataset.test_data,
                                                             list(
                                                                     self.dataset.test_data.keys()),
                                                             self.dataset.train_data,
                                                             top_list, user_emb, item_emb)
            if bool(getattr(self.env.args, 'report_test_modality_subsets', 0)):
                missing_items = self.dataset.test_missing_modality_items['items']
                missing_ratings, full_ratings = evaluation.split_ratings_by_item_membership(
                    self.dataset.test_data, missing_items
                )
                self.last_test_modality_subsets = {}
                for subset_name, subset_ratings in (
                    ('full_modal', full_ratings),
                    ('missing_modal', missing_ratings),
                ):
                    counts = evaluation.rating_counts(subset_ratings)
                    if not subset_ratings:
                        raise ValueError(
                            f'test modality subset {subset_name} has no positive interactions'
                        )
                    subset_hr, subset_recall, subset_ndcg = evaluation.num_faiss_evaluate(
                        subset_ratings,
                        list(subset_ratings.keys()),
                        self.dataset.train_data,
                        top_list,
                        user_emb,
                        item_emb,
                    )
                    self.last_test_modality_subsets[subset_name] = {
                        'counts': counts,
                        'hr': subset_hr,
                        'recall': subset_recall,
                        'ndcg': subset_ndcg,
                    }

        return hr, recall, ndcg, time.time() - t

    def log_test_modality_subsets(self, prefix='test'):
        for subset_name in ('full_modal', 'missing_modal'):
            result = self.last_test_modality_subsets.get(subset_name)
            if not result:
                continue
            counts = result['counts']
            count_message = (
                f'{prefix} subset={subset_name} users={counts["users"]} '
                f'interactions={counts["interactions"]}'
            )
            tool.cprint(count_message)
            if self.env.args.log:
                self.env.test_logger.info(count_message)
            for topk in result['hr']:
                metric_message = (
                    f'{prefix} subset={subset_name} hr@{topk} = {result["hr"][topk]:.5f}, '
                    f'recall@{topk} = {result["recall"][topk]:.5f}, '
                    f'ndcg@{topk} = {result["ndcg"][topk]:.5f}'
                )
                tool.cprint(metric_message)
                if self.env.args.log:
                    self.env.test_logger.info(metric_message)

    @torch.no_grad()
    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch):
        prefix = f'{self.env.args.suffix}_{self.env.args.train_stage}_{self.env.args.penalty_coeff}_epoch'
        model_state_file = os.path.join(self.env.CKPT_PATH, f'{prefix}{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        for file_name in os.listdir(self.env.CKPT_PATH):
            if not file_name.startswith(prefix):
                continue
            file_path = os.path.join(self.env.CKPT_PATH, file_name)
            if file_path != model_state_file and os.path.exists(file_path):
                os.remove(file_path)
