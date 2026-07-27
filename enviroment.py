import logging
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import tool


class Env(object):
    def __init__(self, args):
        self.args = args
        self.DATA_PATH = 'Data' # data path
        self.ROOT_PATH = '' # code path

        self.DATA_PATH = os.path.join(self.DATA_PATH, self.args.dataset)
        self.BASE_PATH = os.path.join(self.ROOT_PATH, 'exp_report')
        self.BASE_PATH = os.path.join(self.BASE_PATH, self.args.dataset)
        self.BOARD_PATH = os.path.join(self.BASE_PATH, 'tensorboard')
        self.BASE_PATH = os.path.join(self.BASE_PATH, self.args.suffix)
        self.CKPT_PATH = os.path.join(self.BASE_PATH, 'ckpt')
        self.LOG_PATH = os.path.join(self.BASE_PATH, 'log')
        self.PIC_PATH = os.path.join(self.BASE_PATH, 'pic')
        self.reset(args)

    def reset(self, args):
        self.args = args
        self.time_stamp = time.strftime('%y-%m-%d-%H', time.localtime(time.time()))
        self._check_direcoty()
        self._init_device()
        self._set_seed(self.args.seed)

        if self.args.log:
            # logging.shutdown()
            self._init_logger()

        if self.args.tensorboard:
            self._init_tensorboard()

    def _check_direcoty(self):
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH, exist_ok=True)
        if not os.path.exists(self.BOARD_PATH):
            os.makedirs(self.BOARD_PATH, exist_ok=True)
        if not os.path.exists(self.CKPT_PATH):
            os.makedirs(self.CKPT_PATH, exist_ok=True)
        if not os.path.exists(self.LOG_PATH):
            os.makedirs(self.LOG_PATH, exist_ok=True)
        if not os.path.exists(self.PIC_PATH):
            os.makedirs(self.PIC_PATH, exist_ok=True)

    def _init_device(self):
        if torch.cuda.is_available() and self.args.use_gpu:
            self.device = torch.device(self.args.device_id)
        else:
            self.device = 'cpu'
        tool.cprint(f'Code is running on {self.device}')

    def _init_logger(self):
        self.train_logger = tool.Log('train', os.path.join(self.LOG_PATH,
                                                              f'{self.time_stamp}_train_log_{self.args.suffix}.log'))
        self.val_logger = tool.Log('val',
                                      os.path.join(self.LOG_PATH, f'{self.time_stamp}_val_log_{self.args.suffix}.log'))
        self.test_logger = tool.Log('test', os.path.join(self.LOG_PATH,
                                                            f'{self.time_stamp}_test_log_{self.args.suffix}.log'))
        self.train_logger.info(self.args)
        self.val_logger.info(self.args)
        self.test_logger.info(self.args)
        tool.cprint(f'Init Logger')


    def _init_tensorboard(self):
        run_dir = os.path.join(self.BOARD_PATH, self.time_stamp + "-" + self.args.suffix)
        hf_repo = getattr(self.args, 'hf_tensorboard_repo', '')
        hf_token = getattr(self.args, 'hf_token', '')
        hf_commit_every = getattr(self.args, 'hf_commit_every', 5)

        if hf_repo:
            if hf_token:
                # HFSummaryWriter internally reads auth from the process environment
                # for some repo metadata requests (e.g. ModelCard loading).
                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            from huggingface_hub import HFSummaryWriter

            writer_kwargs = {
                'repo_id': hf_repo,
                'logdir': run_dir,
                'commit_every': hf_commit_every,
            }
            if hf_token:
                writer_kwargs['token'] = hf_token
            self.w = HFSummaryWriter(**writer_kwargs)
            tool.cprint(f'Init Tensorboard + HF Hub writer ({hf_repo})')
        else:
            self.w = SummaryWriter(run_dir)
            tool.cprint(f'Init Tensorboard')

    def close_env(self):
        if self.args.log:
            logging.shutdown()

        if self.args.tensorboard:
            self.w.close()

    def _set_seed(self, seed):
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
