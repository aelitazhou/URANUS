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

from env import MOESimulator
from para import params

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
sim = MOESimulator(params, datax, dmds, nbrs=None, sorted_grids=sorted_grids)
obs = sim.reset(preds = preds, gridmap = gridmap)
sim.nbrs = None
sim.reset(preds=preds, gridmap=gridmap)
sim.ti = 168   # Starting from the Monday (a work day, and see the difference...)
step_size = len(preds) - sim.ti - 1
#step_size = 288    # twp days
print('Step_size:', step_size)

# No agent
leftovers = []
rewards = []
actions = []
dmd_sum = np.sum(np.array(preds)[:, :, :, 0], axis=0)
print(dmd_sum.shape)
next_loc = np.argsort(np.sum(dmd_sum[:, 0:params['pred_grid']], axis=1))[-params['num_agents']:]

for s in range(step_size):
    t = time.time()
    loc = sim.locs
    action = []
    if s == 0:
        for i in range(params['num_agents']):
            if next_loc[i] == loc[i]:
                action.append(1)
                action.append(next_loc[i])
            else:
                action.append(2)
                action.append(next_loc[i])
    else:
        a_idx = [1 for i in range(params['num_agents'])]
        for i in range(params['num_agents']):
            action.append(a_idx[i])
            action.append(next_loc[i])
    actions.append(action)

    print('Time Step:', sim.ti)
    print('Location:', sim.locs)
    print("Action:", action)
    
    o, r, done, inf = sim.step(action)
    leftovers.append(sim.leftover)
    rewards.append(r)
    actions.append(action)
    
    print('Reward:', r)
    print('Time:', time.time()-t)
    print('-----------------------------')

save_var = str(params['penalty_w']) + '_' + str(params['x_cost']) + '_' + str(params['y_cost']) + '_' + str(params['i_cost']) + '_' + str(params['pred_grid']) + '_' + str(params['num_agents'])

np.save('RLA/ST_' + save_var + '_R', rewards)
np.save('RLA/ST_' + save_var + '_L', leftovers)
np.save('RLA/ST_' + save_var + '_A', actions)




