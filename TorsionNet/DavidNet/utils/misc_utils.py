#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import random 
import pickle
import os
import datetime
import torch
import time
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def random_seed(seed=None):
    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False 

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def select_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        if gpu_id >= 0:
            Config.DEVICE = torch.device('cuda:%d' % (gpu_id))
        else:
            Config.DEVICE = torch.device('cpu')

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x

def to_np(t):
    return t.cpu().detach().numpy()