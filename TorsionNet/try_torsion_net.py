import numpy as np
import random
import argparse
import torch
import logging

import gym

from main.config import Config
from main.environments import DummyVecEnv, OriginalReturnWrapper

# New Shit
from DavidNet import GibbsMolEnv, PruningSetLogGibbs
from DavidNet.agent import PPORecurrentAgent
from DavidNet.network import *
from DavidNet.utils import random_seed

from scm.plams import Molecule, to_rdmol

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on the device: ", device)

# file_ext, folder_name = "json", "lignin_hightemp/"
# file_ext, folder_name = "xyz", "../ams_mols/n_butane/"

# cfg_env = Config()
# cfg_env.folder_name, cfg_env.file_ext = "../ams_mols/n_butane/", "xyz"
# cfg_env.plams_mol = False
# cfg_env.plams_tor = False

# cfg_env.gibbs_normalize = True
# cfg_env.temp_normal = 1.0
# cfg_env.sort_by_size = True
# cfg_env.pruning_thresh = 0.05
# cfg_env.eval = False


# GibbsMolEnv(cfg_env)


class LigninEnv(PruningSetLogGibbs):
    def __init__(self, cfg):
        super(LigninEnv, self).__init__(cfg)


class Curriculum():           
    def __init__(self, win_cond=0.7, success_percent=0.7, fail_percent=0.2, min_length=100):
        self.win_cond = win_cond
        self.success_percent = success_percent
        self.fail_percent = fail_percent
        self.min_length = min_length

    def return_win_cond():
        return self.win_cond


### CREATING INIT FUNCTION USING TWO CFG OBJECTS.
def init_ppo_agent():
    cfg_ppo, cfg_env = Config(), Config()
    
    # Creating config containing all information and parameters for environment.
    cfg_env.folder_name, cfg_env.file_ext = "lignin_hightemp/", "json"
    # cfg_env.folder_name, cfg_env.file_ext = "../ams_mols/n_butane/", "xyz"
    cfg_env.gibbs_normalize = True
    cfg_env.temp_normal = 1.0
    cfg_env.sort_by_size = False
    cfg_env.pruning_thresh = 0.05
    cfg_env.eval = False 

    cfg_env.plams_mol = False
    cfg_env.plams_tor = False

    # Creating config containing all models and information for ppo agent.
    cfg_ppo.num_workers = 1
    single_process = (cfg_ppo.num_workers == 1)
    
    cfg_ppo.linear_lr_scale = False

    base_lr = 5e-6
    if cfg_ppo.linear_lr_scale:
        lr = base_lr * cfg_ppo.num_workers
    else:
        lr = base_lr * np.sqrt(cfg_ppo.num_workers)

    cfg_ppo.curriculum = Curriculum(min_length=cfg_ppo.num_workers)
    
    # cfg_ppo.train_env = DummyVecEnv([OriginalReturnWrapper(GibbsMolEnv(cfg_env))])
    cfg_ppo.train_env = LigninEnv(cfg_env)
    # cfg_ppo.train_env = PruningSetGibbs(cfg_env)
    cfg_ppo.network = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    cfg_ppo.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    cfg_ppo.hidden_size = cfg_ppo.network.dim
    cfg_ppo.discount = 0.9999
    cfg_ppo.use_gae = True
    cfg_ppo.gae_tau = 0.95
    cfg_ppo.value_loss_weight = 0.25 # vf_coef
    cfg_ppo.entropy_weight = 0.001
    cfg_ppo.gradient_clip = 0.5
    cfg_ppo.rollout_length = 20
    cfg_ppo.recurrence = 5
    cfg_ppo.optimization_epochs = 4
    # cfg_ppo.mini_batch_size = cfg_ppo.rollout_length * cfg_ppo.num_workers
    cfg_ppo.mini_batch_size = 25
    cfg_ppo.ppo_ratio_clip = 0.2
    cfg_ppo.save_interval = cfg_ppo.num_workers * 1000 * 2
    cfg_ppo.eval_interval = cfg_ppo.num_workers * 1000 * 2
    cfg_ppo.eval_episodes = 1
    # cfg_ppo.eval_env = Task('LigninPruningSkeletonValidationLong-v0', seed=random.randint(0,7e4))

    return PPORecurrentAgent(cfg_ppo)


random_seed(4)
agent = init_ppo_agent()
agent.network.to(device)

agent.run_steps()
