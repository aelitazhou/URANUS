import os, sys, time, random, json, pickle, itertools
import numpy as np

import torch, torchvision
from torch.autograd import Variable
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from model import MLP

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



# BLSTM
model = MLP()

for n, params in model.named_parameters():
    print(n, ':', params.size())
total_b = sum(p.numel() for p in model.parameters())
print('Total:', total_b)

