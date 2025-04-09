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

from env_sdp_map import MOESimulator
from para import params
from model import MLP 

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
demands = demands[:, 144*15:]
predictions = predictions[144*15:, :, :, :]
print('Demands shape:', demands.shape)       # demands[:, 144:-11] correponding to predictions
print('Predictions shape:', predictions.shape)

# experiments
preds = list(predictions)
dmds = list(demands)
gridmap = np.load('../data/gridmap.npy', allow_pickle=True).item()
print('Predictions, demands and gridmap loaded')
'''
grid_map = np.array([[0., 1., 1., 1., 1., 2., 1., 2., 3., 2.],
                     [1., 0., 2., 1., 1., 2., 1., 3., 3., 2.],
                     [1., 2., 0., 2., 1., 2., 1., 2., 2., 2.],
                     [1., 1., 2., 0., 1., 2., 2., 3., 3., 2.],
                     [1., 1., 1., 1., 0., 2., 2., 3., 3., 2.],
                     [2., 2., 2., 2., 2., 0., 1., 2., 2., 1.],
                     [1., 1., 1., 2., 2., 1., 0., 2., 2., 1.],
                     [2., 3., 2., 3., 3., 2., 2., 0., 1., 2.],
                     [3., 3., 2., 3., 3., 2., 2., 1., 0., 2.],
                     [2., 2., 2., 2., 2., 1., 1., 2., 2., 0.]])
'''

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
'''
grid_map = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 0., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 0., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.]])
'''

# greedy DP
def dp_agent(sim, params):
    # init
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
     
    # dp process
    for a in dp_idxs: 
        # backward reward cumulation
        #print('initialization--------------')
        action = None
        m_map = np.zeros((n_g, n_t)) # memo for road
        r_map = np.zeros((n_g, n_t))   # cumulative reward map
        ini = [max(sim.dp_stay(sim.ti, n_t-1, g, agent_map[g][n_t-1]), sim.dp_idle()) for g in range(n_g)]
        ini_ = [i*(params['discount_factor']**(n_t-1)) for i in ini]
        r_map[:, -1] = ini_
        #print('agent', a, 'ini', ini)
         
        # n_t == 1, static agent; n_t != 1, dp agent
        if n_t != 1:
            for i in reversed(range(n_t-1)):    # i in [0,10]
                for j in range(n_g):
                    if i == 0 and j != sim.locs[a]:  # at time slot 1, loc=sim.locs[a]
                        continue
                    temp = float('-inf')
                    prev = None
                    for k in range(n_g):
                        # stay-serve/idle
                        if j == k:
                            score_ = [sim.dp_idle(), sim.dp_stay(sim.ti, i, k, agent_map[k][i])]  
                            score = max(score_)
                            score_d = score * (params['discount_factor']**i)
                            if r_map[k][i+1] + score_d > temp:
                                temp = r_map[k][i+1] + score_d
                                prev = k
                                action = score_.index(score)
                        
                        # move
                        else:
                            # grid_map[j][k] is the time slot distance between j and k
                            ts = grid_map[j][k]
                         
                            if ts+i <= (n_t-1): # reachable within dp process
                                pre_reward = r_map[k][int(i+ts)]
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
            score_ = [sim.dp_idle(), sim.dp_stay(sim.ti, 0, sim.locs[a], agent_map[sim.locs[a]][0])]  # stay-serve/idle/move
            score = max(score_)
            action = score_.index(score)
            if action == 0 or action == 1:
                prev = sim.locs[a]
            else:
                left_locs = np.where(grid_map[sim.locs[a]]==1)[0] # time slot=1 locs
                prev = random.choices(left_locs, k=1)
            m_map[sim.locs[a]][0] = prev
            agent_map[sim.locs[a]][0] +=1
        


        actions[2*a] = action
        actions[2*a+1] = prev
        #actions.append(action)
        #actions.append(prev)
    
        #print(m_map)
        #print(r_map)
        #print(agent_map) 
        #print(actions)
        
    return actions   # next time slot actions and locations [a, l, a, l, a, l....]



# experiment
sim = MOESimulator(params, datax, dmds, grid_map, nbrs=None, sorted_grids=sorted_grids)
sim.nbrs = None
sim.reset(preds=preds, gridmap=gridmap)
sim.ti = 24  # Starting from 12/1/2013 Monday 4 a.m. 
#sim.ti = 3060
step_size = len(preds) - sim.ti - 1
print('Step_size:', step_size)

leftovers = []
rewards = []
actions = []
penalties = []
utilities = []
cvar_x = []
cvar_y = []
ttt = time.time()
for s in range(step_size): 
    tt = time.time()

    action = dp_agent(sim, params)
    new_action = sim.reloc(action)
    
    print('Time Step:', sim.ti)
    print('location:', sim.locs)
    print('Action:',  action)
    print('New_Action:',  new_action)
    
    s_, r, d, _ = sim.step(new_action)
    

    actions.append(new_action)
    leftovers.append(sim.leftover)
    rewards.append(r)
    penalties.append(sim.penalty)
    utilities.append(sim.util)
    
    #print('Reward:', r)
    #print('Time:', time.time()-tt, flush=True)   # 8.2
    #print('-----------------------------')
print('Time:', time.time()-ttt)

#save_var = str(params['pred_grid']) + '_' + str(params['num_agents']) + '_' + str(params['eps'])
save_var = str(params['penalty_w']) + '_' + str(params['pred_grid']) + '_' + str(params['num_agents'])
#save_var = str(params['pred_grid']) + '_' + str(params['num_agents'])

np.save('RLAUP/SDP_M_' + save_var + '_R', rewards)
np.save('RLAUP/SDP_M_' + save_var + '_L', leftovers)
np.save('RLAUP/SDP_M_' + save_var + '_A', actions)
np.save('RLAUP/SDP_M_' + save_var + '_U', utilities)
np.save('RLAUP/SDP_M_' + save_var + '_P', penalties)
#np.save('lc_sdp/cvar_x_' + save_var, cvar_x)
#np.save('lc_sdp/cvar_y_' + save_var, cvar_y)
print('files saved')

