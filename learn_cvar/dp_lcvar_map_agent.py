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

from env_lcvar_map import MOESimulator
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

demands = np.load('../data/Blstm10_dmds.npy')[:, 144:-11]/params['demand_scaling_ratio']
predictions = np.load('../data/Blstm10_pred.npy')/params['demand_scaling_ratio']
print('Demands shape:', demands.shape)       # demands[:, 144:-11] correponding to predictions
print('Predictions shape:', predictions.shape)

# experiments
preds = list(predictions)
dmds = list(demands)
gridmap = np.load('../data/gridmap.npy', allow_pickle=True).item()
print('Predictions, demands and gridmap loaded')

grid_map = np.array([[0., 1., 1., 1., 1., 1., 1., 2., 2., 1.],
       [1., 0., 1., 1., 1., 1., 1., 2., 2., 1.],
       [1., 1., 0., 1., 1., 1., 1., 1., 2., 2.],
       [1., 1., 1., 0., 1., 2., 1., 2., 2., 2.],
       [1., 1., 1., 1., 0., 2., 1., 2., 2., 2.],
       [1., 1., 1., 2., 2., 0., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 0., 1., 2., 1.],
       [2., 2., 1., 2., 2., 1., 1., 0., 1., 2.],
       [2., 2., 2., 2., 2., 1., 2., 1., 0., 2.],
       [1., 1., 2., 2., 2., 1., 1., 2., 2., 0.]])

# experiment
sim = MOESimulator(params, datax, dmds, grid_map, nbrs=None, sorted_grids=sorted_grids)
sim.nbrs = None
sim.reset(preds=preds, gridmap=gridmap)
sim.ti = 0   # Starting from the Monday (a work day, and see the difference...)
step_size = 144*15     # use 15 days to train mlp tp approach cvar
print('Step_size:', step_size)


# greedy DP

def dp_agent(sim, params):
    n_g = sim.num_grids
    n_t = params['pred_grid']
    n_a = params['num_agents']
    agent_map = np.ones((n_g, n_t)) # cumulative agent road map
    actions = np.zeros((2*n_a, 1))

    # only do dp process for those not in the moving process
    f_idxs = [i for i in range(n_a)]
    m_idxs = []      # locs for moving process
    for i, j in enumerate(sim.locs):
        if j == 100:
            m_idxs.append(i)
            actions[2*i] = 2
            actions[2*i+1] = 100
    # moving process also take the destination location
    for l in m_idxs:
        agent_map[sim.des_locs[l]][0] += 1

    dp_idxs = list(set(f_idxs) - set(m_idxs))   # locs for dp process
    
    cvars_x, cvars_y = [], []
    for a in dp_idxs:
        # backward reward cumulation
        #print('initialization--------------')
        action = None
        m_map = np.zeros((n_g, n_t)) # memo for road
        r_map = np.zeros((n_g, n_t))   # cumulative reward map
        ini = [max(sim.dp_stay(sim.ti, n_t-1, g, agent_map[g][n_t-1]), sim.dp_idle()) for g in range(n_g)]
        r_map[:, -1] = ini

        if n_t != 1:
            # for t = [sim.ti+1, sim.ti+12]
            cvar_x, cvar_y = [], []
            for i in reversed(range(n_t-1)):    # i in [0,10]
                cc_x, cc_y, cc_x_p, cc_y_p = [], [], [], []
                for j in range(n_g):
                    if i == 0 and j != sim.locs[a]:  # at time slot 1, loc=sim.locs[a]
                        continue
                    temp = float('-inf')
                    prev = None
                    for k in range(n_g):
                        if j == k:
                            score_ = [sim.dp_idle(), sim.dp_stay(sim.ti, i, k, agent_map[k][i])]  # stay-serve/idle
                            score = max(score_)
                            score_d = score * (params['discount_factor']**i)
                            # only dp road
                            '''
                            cc_x.append(sim.cvar_x)
                            cc_y.append(sim.cvar_y)
                            cc_x_p.append(sim.cvar_x_p)
                            cc_y_p.append(sim.cvar_y_p)
                            '''
                            if r_map[k][i+1] + score_d > temp:
                                temp = r_map[k][i+1] + score_d
                                prev = k
                                action = score_.index(score)
           
                        else:
                            # grid_map[j][k] is the time slot distance between j and k
                            ts = grid_map[j][k]

                            if ts+i <= (n_t-1): # reachable within dp process
                                pre_reward = r_map[k][int(i+ts)] * (params['discount_factor']**(ts-1))
                                score = sim.dp_move() * grid_map[j][k]        # move_cost * number of time slot
                                score_d = score * (params['discount_factor']**i)
                            else:
                                pre_reward = float('-inf')

                            if pre_reward + score_d > temp:
                                temp = pre_reward + score_d
                                prev = k
                                action = 2
                                for s in range(int(ts)): # memo for the agent that needs to move multiple hops
                                    m_map[j][i - s] = k
                    m_map[j][i] = prev
                    r_map[j][i] = temp
                    '''
                    if action != 2:
                        cvar_x.append(sim.cvar_x)
                        cvar_y.append(sim.cvar_y)
                        cvar_x.append(sim.cvar_x_p)
                        cvar_y.append(sim.cvar_y_p)
                        '''
            #if ts == 1:   # not in loc=100, ca n move to k directly
            # whether in loc=100 or not, agent_map need to record the agent destination for the other agents
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
        '''
        cvars_x.append(cvar_x)
        cvars_y.append(cvar_y)
        '''
        actions[2*a] = action
        actions[2*a+1] = prev
        #actions.append(action)
        #actions.append(prev)
        '''
        print(m_map)
        print(r_map)
        print(agent_map) 
        print(actions)
        '''
    #return actions, cvars_x, cvars_y   # next time slot actions and locations [a, l, a, l, a, l....]
    return actions   # next time slot actions and locations [a, l, a, l, a, l....]



cvars_x = []
cvars_y = []
for s in range(step_size): 
    tt = time.time()

    #action, cvar_x, cvar_y = dp_agent(sim, params)
    action = dp_agent(sim, params)
    new_action = sim.reloc(action)

    print('Time Step:', sim.ti)
    print('location:', sim.locs)
    print('Action:',  action)
    print('New Action:', new_action)
    
    s_, r, d, _ = sim.step(action)
    # last step of dp
    cvars_x.append(sim.cvar_x)
    cvars_y.append(sim.cvar_y)
    cvars_x.append(sim.cvar_x_p)
    cvars_y.append(sim.cvar_y_p)
    '''
    for i in range(params['num_agents']):
        for j in cvar_x[i]:
            cvars_x.append(j)
        for k in cvar_y[i]:
            cvars_y.append(k)
    len_x = len(cvar_x[0]) + len(cvar_x[1]) + len(cvar_x[2])
    print(len_x)
    '''
    '''
    cvars_x.append(cvar_x)
    cvars_y.append(cvar_y)
    '''
    print('Time:', time.time()-tt, flush=True)   # 8.2
    print('-----------------------------')

save_var = str(params['pred_grid']) + '_' + str(params['num_agents']) 

np.save('cvar_data/cvar_x_' + save_var, cvars_x)
np.save('cvar_data/cvar_y_' + save_var, cvars_y)




