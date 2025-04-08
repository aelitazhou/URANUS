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
data = pd.read_csv('../data/mi.csv')
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
sorted_grids = np.loadtxt('../data/gridx_10')
datax = data.loc[ data['grid'].isin(sorted_grids) ].reset_index(drop=True)
print('Total data size:', datax.shape)

demands = np.load('../data/Blstm10_dmds.npy')[:, 144:]/params['demand_scaling_ratio']
predictions = np.load('../data/Blstm10_pred.npy')/params['demand_scaling_ratio']
print('Demands shape:', demands.shape)       # demands[:, 144:-11] correponding to predictions
print('Predictions shape:', predictions.shape)

# experiments
preds = list(predictions)
dmds = list(demands)
gridmap = np.load('../data/gridmap.npy', allow_pickle=True).item()
print('Predictions, demands and gridmap loaded')

# experiment
sim = MOESimulator(params, datax, dmds, nbrs=None, sorted_grids=sorted_grids)
#obs = sim.reset(preds = preds, gridmap = gridmap)
sim.nbrs = None
sim.reset(preds=preds, gridmap=gridmap)
sim.ti = 0  
step_size = 144*15     # use 15 days to train mlp tp approach cvar
print('Step_size:', step_size)


# greedy DP

def dp_agent(sim, params):
    n_g = sim.num_grids
    n_t = params['pred_grid']
    n_a = params['num_agents']
    agent_map = np.ones((n_g, n_t)) # cumulative agent road map
    actions = []
    
    dpss_x = [] 
    dpss_y = []        
    for a in range(n_a):
        # backward reward cumulation
        #print('initialization--------------')
        action = None
        m_map = np.zeros((n_g, n_t)) # memo for road
        r_map = np.zeros((n_g, n_t))   # cumulative reward map
        rr_map = np.zeros((n_g, n_t))  # reward map 
        
        dps_x = np.zeros((4, n_g, n_t-2)) # score, mean, std, z
        dps_y = np.zeros((n_g, n_t-2))
        
        r_ini = []
        rr_ini = []
        for g in range(n_g):
            score_ = [sim.dp_stay(sim.ti, n_t-1, g, agent_map[g][n_t-1]), sim.dp_idle()]
            score_r = max(score_)
            index = score_.index(score_r)
            if index == 0:
                score_rr = score_r - sim.dp_stay_cost()
            else:
                score_rr = score_r - sim.dp_idle()
            r_ini.append(score_r)
            rr_ini.append(score_rr)
        r_map[:, -1] = r_ini
        rr_map[:, -1] = rr_ini
        if n_t != 1:
            for i in reversed(range(n_t-1)):    # i in [0,10]
                for j in range(n_g):
                    if i == 0 and j != sim.locs[a]:  # at time slot 0 (0-11), loc=sim.locs[a]
                        continue
                    temp = float('-inf')
                    temp_r = float('-inf')
                    prev = None
                    for k in range(n_g):
                        if j == k:
                            score_ = [sim.dp_idle(), sim.dp_stay(sim.ti, i, k, agent_map[k][i])]  # stay-serve/idle
                            score = max(score_)
                            # rr_map
                            if score_.index(score) == 0:
                                score_rr = score - sim.dp_idle()
                            else:
                                score_rr = score - sim.dp_stay_cost()
                            rr_map[k][i] = score_rr
                           
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
        
        dps_x = rr_map[:, 1:]
        dps_y = r_map[:, 1]
        
        dpss_x.append(dps_x)
        dpss_y.append(dps_y)
        actions.append(action)
        actions.append(prev)
    
    return actions, np.array(dpss_x), np.array(dpss_y)   # next time slot actions and locations [a, l, a, l, a, l....]

dps_x = []
dps_y = []
#step_size = 2
for s in range(step_size): 
    tt = time.time()
    print('Time Step:', sim.ti)
    action, dpss_x, dpss_y  = dp_agent(sim, params) 
    s_, r, d, _ = sim.step(action)
    
    dps_x.append(dpss_x)
    dps_y.append(dpss_y)
    
    print('Time:', time.time()-tt, flush=True)   # 8.2
    print('-----------------------------')
dps_x = np.array(dps_x).reshape(step_size*params['num_agents'], sim.num_grids, params['pred_grid']-1)
dps_y = np.array(dps_y).reshape(step_size*params['num_agents'], sim.num_grids)
print(dps_x.shape, dps_y.shape)

save_var = str(params['pred_grid']) + '_' + str(params['num_agents']) 

np.save('ldp_train_data/dps_x_' + save_var, dps_x)
np.save('ldp_train_data/dps_y_' + save_var, dps_y)




