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

from env_ldp import MOESimulator
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

demands = np.load('data/Blstm10_dmds.npy')[:, 144:-11]/params['demand_scaling_ratio']
predictions = np.load('data/Lstm10_pred.npy')/params['demand_scaling_ratio']
demands = demands[:, 144*15:]
predictions = predictions[144*15:, :, :]
print('Demands shape:', demands.shape)       # demands[:, 144:-11] correponding to predictions
print('Predictions shape:', predictions.shape)

# experiments
preds = list(predictions)
dmds = list(demands)
gridmap = np.load('data/gridmap.npy', allow_pickle=True).item()
print('Predictions, demands and gridmap loaded')

# greedy DP
def dp_agent(sim, params):
    n_g = sim.num_grids
    n_t = params['pred_grid']
    n_a = params['num_agents']
    agent_map = np.ones((n_g, n_t)) # cumulative agent road map
    actions = []
    
    for a in range(n_a): 
        # backward reward cumulation
        #print('initialization--------------')
        action = None
        m_map = np.zeros((n_g, n_t)) # memo for road
        r_map = np.zeros((n_g, n_t))   # cumulative reward map
        ini = [max(sim.dp_stay(sim.ti, n_t-1, g, agent_map[g][n_t-1]), sim.dp_idle()) for g in range(n_g)]
        r_map[:, -1] = ini
        #print('agent', a, 'ini', ini)
        
        if n_t != 1:
            # for t = [sim.ti+1, sim.ti+12]
            for i in reversed(range(n_t-1)):    # i in [0,10]
                for j in range(n_g):
                    if i == 0 and j != sim.locs[a]:  # at time slot 1, loc=sim.locs[a]
                        continue
                    temp = float('-inf')
                    prev = None
                    for k in range(n_g):
                        if j == k:
                            score_ = [sim.dp_idle(), sim.dp_stay(sim.ti, i, k, agent_map[k][i])]  # stay-serve/idle
                            score = max(score_)
                            if r_map[k][i+1] + score > temp:
                                temp = r_map[k][i+1] + score
                                prev = k
                                action = score_.index(score)
                        else:
                            score = sim.dp_move()        # move
                            if r_map[k][i+1] + score > temp:
                                temp = r_map[k][i+1] + score
                                prev = k
                                action = 2
                    m_map[j][i] = prev
                    r_map[j][i] = temp

            # agent map, at this time, m_map[sim.locs[a]][0] = prev
            step = prev
            agent_map[sim.locs[a]][0] += 1
            agent_map[step][1] += 1
            for n in range(1, n_t-1):     # n in [1, 10]
                step = m_map[int(step)][n]
                agent_map[int(step)][n+1] += 1

        else:     # n_t = 1, only one time slot is available, no dp
            score_ = [sim.dp_idle(), sim.dp_stay(sim.ti, 0, sim.locs[a], agent_map[sim.locs[a]][0]), sim.dp_move()]  # stay-serve/idle/move
            score = max(score_)
            action = score_.index(score)
            if action == 0 or action == 1:
                prev = sim.locs[a]
            else:
                left_locs = list(range(10))
                left_locs.remove(sim.locs[a])
                prev = random.choice(left_locs, k=1)
            m_map[sim.locs[a]][0] = prev
            agent_map[sim.locs[a]][0] +=1

        actions.append(action)
        actions.append(prev)
        '''
        print(m_map)
        print(r_map)
        print(agent_map) 
        print(actions)
        '''
    return actions   # next time slot actions and locations [a, l, a, l, a, l....]


# experiment
sim = MOESimulator(params, datax, dmds, nbrs=None, sorted_grids=sorted_grids)
sim.nbrs = None
sim.reset(preds=preds, gridmap=gridmap)
sim.ti = 24  # Starting from 12/1/2013 Monday 4 a.m. 
step_size = len(preds) - sim.ti - 1
print('Step_size:', step_size)

leftovers = []
rewards = []
actions = []
penalties = []
utilities = []

for s in range(step_size): 
    tt = time.time()

    action = dp_agent(sim, params)
    
    print('Time Step:', sim.ti)
    #print('location:', sim.locs)
    #print('Action:',  action)
    
    s_, r, d, _ = sim.step(action)

    actions.append(action)
    leftovers.append(sim.leftover)
    rewards.append(r)
    penalties.append(sim.penalty)
    utilities.append(sim.util)

    #print('Reward:', r)
    print('Time:', time.time()-tt, flush=True)   # 8.2
    print('-----------------------------')

save_var = str(params['penalty_w']) + '_' + str(params['pred_grid']) + '_' + str(params['num_agents'])
#save_var = str(params['pred_grid']) + '_' + str(params['num_agents'])

np.save('RLAUP/LDP_' + save_var + '_R', rewards)
np.save('RLAUP/LDP_' + save_var + '_L', leftovers)
np.save('RLAUP/LDP_' + save_var + '_A', actions)
np.save('RLAUP/LDP_' + save_var + '_U', utilities)
np.save('RLAUP/LDP_' + save_var + '_P', penalties)

