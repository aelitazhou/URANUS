import os, sys, time, random, json, pickle
import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp
import re
random.seed(12345678)
np.random.seed(12345678)

import torch, torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

import geopandas as gpd
import geopy.distance
import shapely
from shapely.geometry import Point, Polygon

import gym
from gym import Env, spaces

from env_drl import MOESimulator
from para import params

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class SimCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, fn='drl_train_log.log', append=False):
        super(SimCallback, self).__init__(verbose)
        self.fn = fn
        self.append = append
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.out = open(self.fn, ('a' if self.append else 'w'))
        # pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        envs = self.locals['env']
        outstr = '--------------------------------------------\n'
        outstr += ('Time:' + str(envs.get_attr('ti')[0]-1) + '\n\n')
        outstr += ('Action:' + str(envs.get_attr('last_action')[0]) + '\n\n')
        outstr += ('Utils:' + str(envs.get_attr('utils')[0]) + '\n\n')
        outstr += ('Penalties:' + str(envs.get_attr('penalties')[0]) + '\n\n')
        outstr += ('Costs:' + str(envs.get_attr('costs')[0]) + '\n\n')
        outstr += ('Reward:' + str(envs.get_attr('reward')[0]) + '\n')
        self.out.write(outstr)
        self.out.flush()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.out.close()
        # pass


seed = params['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# set the init function of the worker(s) to be fed to the Dataloader
#def _init_fn(worker_id):
#    np.random.seed(int(seed)

t = time.time()
print('Reading mi.csv')
data = pd.read_csv('data/mi.csv')
print('Data shape:', data.shape)
print('Reading time:', time.time()-t)


def reindex_arb(data, xy = (100, 100), xycell = (4, 4), lrbt = (0, 100, 0, 100)):
    '''
    Reindex milano grids arbitrarily based on three parameters.

    xy = (x, y): total shape of the grids
    xycell = (xcell, ycell): shape of each cluster of cells to be aggregated
    lrbt = (l, r, b, t): left, right, bottom, top boundaries of the xy shape to be considered
    '''
    # 1. Reorganize all grids
    grids = data['grid'].unique()
    gs = grids.reshape(xy)

    # 2. Take only slices defined by lrbt
    l, r, b, t = lrbt
    gs = gs[b:t, l:r]
    x, y = gs.shape

    # 3. Slice by xycells
    xcell, ycell = xycell
    xrng = list(range(0, x, xcell))
    yrng = list(range(0, y, ycell))
    slices = []
    for xx in xrng:
        for yy in yrng:
            idxs = gs[xx:xx+xcell, yy:yy+ycell]
            slices.append(np.reshape(idxs, -1))

    # 4. Organize idmap
    idmap = { gid: -1 for gid in data['grid'].unique() }
    for i, slc in enumerate(slices):
        for gid in slc:
            idmap[gid] = i

    # 5. Reorganize data
    data1 = data.copy()
    data1['grid'] = data1['grid'].transform(lambda x: idmap[x])
    if 'time' in data1.columns:
        data1 = data1.groupby(['time', 'grid']).sum().reset_index()
    else:
        data1 = data1.dissolve(by='grid').reset_index()
    data1 = data1[ data1['grid'] != -1 ]

    # Return slices and idmap
    return data1, slices, idmap


xy = (100, 100)
xycell = (4, 4)
lrbt = (0, 100, 0, 100)
data, slices, idmap = reindex_arb(data, xy=xy, xycell=xycell, lrbt=lrbt)

# Find 10 grid with max total demands
sorted_grids = np.loadtxt('data/gridx_10')
datax = data.loc[ data['grid'].isin(sorted_grids) ].reset_index(drop=True)
print('Total data size:', datax.shape)

demands = np.load('data/Blstm12_dmds.npy')[:, 144:-11]/params['demand_scaling_ratio']
predictions = np.load('data/Blstm12_preds.npy')/params['demand_scaling_ratio']
print('Demands shape:', demands.shape)       # demands[:, 144:-11] correponding to predictions
print('Predictions shape:', predictions.shape)

# experiments
preds = list(predictions)
dmds = list(demands)
gridmap = np.load('data/gridmap.npy', allow_pickle=True).item()
print('Predictions, demands and gridmap loaded')


# experiment
env = DummyVecEnv([lambda: MOESimulator(params, datax, dmds, nbrs=None, sorted_grids=sorted_grids, preds=preds, gridmap=gridmap)])
model_fn = 'a2cw_qns4_' + str(params['pred_grid']) + '_' +  str(params['num_agents'])
drl_model = A2C("MultiInputPolicy", env, verbose=1, n_steps=6, learning_rate=0.0001)

# Train for how many days (start from 4a.m.), and each day for how many times
n_day = 29
n_time = 1000
# DRL agent
print("---------------------------------------")
print("DRL Agent Training")

# RL for training day by day, action space [a, l, a, l, a, l]
# All_RL for training all days together, action space [a, l, a, l, a, l]
# Action_RL for training using action space [l, l, l]

order = []
for i in range(n_time):
    a = list(range(n_day))
    random.shuffle(a)
    order.append(a)
order = np.array(order)

t = time.time()
# Train
for itr in range(n_time):
    for day in range(n_day):
        drl_model.env.set_attr('start_ti', 144*order[itr][day]+168)
        drl_model.env.set_attr('end_ti', 144*(order[itr][day]+1)+168)
        drl_model.learn(total_timesteps=144, callback=SimCallback(fn=model_fn+'.log', append=True))
        #drl_model.save( 'a2cw/' + model_fn + ( '_%d_%d' % (day, itr) ) )
print('Training time:', time.time()-t)
drl_model.save( 'a2cw/' + model_fn )

print("---------------------------------------")
print("DRL Agent Testing")
drl_model.load( 'a2cw/' + model_fn)

leftovers = []
rewards = []
actions = []
sim = MOESimulator(params, datax, dmds, nbrs=None, sorted_grids=sorted_grids, preds=preds, gridmap=gridmap)
sim.start_ti = 168
obs = sim.reset(preds = preds, gridmap = gridmap)
step_size = len(preds) - sim.ti - 1
#step_size = 144*n_day
t = time.time()
for _ in range(step_size):
    
    print('Time Step:', sim.ti)
    action, _states = drl_model.predict(obs)
    obs, r, dones, info = sim.step(action)

    leftovers.append(sim.leftover)
    rewards.append(r)
    actions.append(sim.last_action)

    print('Location:', sim.locs)
    print("Action:", sim.last_action) 
    print('Reward:', r)
    print('-----------------------------')
print('Testing time:', time.time()-t)

save_var = str(params['penalty_w']) + '_' + str(params['x_cost']) + '_' + str(params['y_cost']) + '_' + str(params['i_cost']) + '_' + str(params['pred_grid']) + '_' + str(params['num_agents'])

np.save('RLA/A2CW_' + save_var + '_R', rewards)
np.save('RLA/A2CW_' + save_var + '_L', leftovers)
np.save('RLA/A2CW_' + save_var + '_A', actions)




