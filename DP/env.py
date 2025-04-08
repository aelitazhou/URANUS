import os, sys, time, random, json, pickle
import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp
import re
random.seed(12345678)
np.random.seed(12345678)

import torch, torchvision
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from model import BLSTM

import geopandas as gpd
import geopy.distance
import shapely
from shapely.geometry import Point, Polygon

import gym
from gym import Env, spaces
from para import params


class MOESimulator(Env):
    """A Moving Edge simulator based on OpenAI gym"""
    metadata = {'render.modes': ['human']}


    ###################################################################################
    # Initialization
    ###################################################################################

    # Function to calculate types of grids
    def calc_grid_types(self, data):
        '''Calculate type of each grid ID, based on sum demands in different periods.'''
        data1 = data.copy()
        t_st = data1['time'].min()
        data1['ts'] = (((data1['time'] - t_st)/600000).astype(int)%144).between(4*6, 16*6, inclusive='left')
        dfgb = data1.groupby(['grid', 'ts']).max().reset_index()
        dfgb1 = dfgb[ dfgb['ts'] == True ].set_index('grid')
        dfgb2 = dfgb[ dfgb['ts'] == False].set_index('grid')

        gtype = {}
        for grid in dfgb['grid'].unique():
            gtype[grid] = int( dfgb1.loc[ grid ]['internet'] > dfgb2.loc[ grid ]['internet'] )

        return gtype

    def __init__(self, params, data, dmds, model, nbrs = None, sorted_grids = None, preds = None, gridmap = None):
        '''[TODO]: Some grids have missing timestamps possibly due to 0 activity. Should make up for those and fill the 0 values. For now we just use the top 50 grids which do not have this issue.'''
        super(MOESimulator, self).__init__()

        # Reset seeds
        random.seed(params['seed'])
        np.random.seed(params['seed'])

        # Initialize parameters
        self.params = params
        self.num_agents = params['num_agents']
        self.num_grids = params['num_grids']



        # Store demands data
        self.data = data.copy()                          # all time-series data, with key 'time' indexing time slot, key 'grid' indexing grid, and key 'internet' storing the demand
        self.data['internet'] /= params['demand_scaling_ratio']     # scale demands such that they are of a relatively small scale
        self.times = list(data['time'].unique())  # all time slots (including initial feature time slots of length params['predict_input_size'])
        self.grids = list(data['grid'].unique())  # all grids
        if sorted_grids is not None:
            self.grids = sorted_grids

        #self.demands = [ self.data[ self.data['grid'] == grid ].reset_index(drop=True)['internet'].to_numpy() for i, grid in enumerate(self.grids) ]    # array of internet activities, ordered in time slots
        self.demands = dmds

        self.t_start = params['predict_input_size']# Starting time slot to make a prediction/decision
        self.t_end = len(self.times)               # Ending time slot to make a prediction/decision
        self.t_num = self.t_end - self.t_start     # Number of time slots to consider

        self.nbrs = nbrs           # Neighbor of grids, indexed by grid ID

        self.grid_types = self.calc_grid_types(self.data)
        
        self.model = model
        self.preds = None
        self.gridmap = None

        if (preds is not None) and (gridmap is not None):
            self.preds = preds
            self.gridmap = gridmap


        # Store util/panelty/cost
        self.costs = list(np.zeros(self.num_agents))
        self.penalties = list(np.zeros(len(self.grids)))
        self.utils = list(np.zeros(len(self.grids)))



        # Store start and end time slots
        self.start_ti = params['predict_input_size']
        self.end_ti = len(self.times)


        ####################################
        # Baseline rewards
        self.baseline_rewards = [ None for t in self.times ]               # baseline rewards: initialized to None
        self.baseline_action = list(np.concatenate( [ [0, i] for i in range(self.num_agents) ] ))   # baseline action: IDLE at the top K grids
        self.baseline_locs = [ i for i in range(self.num_agents) ]                                  # baseline locations of agents



        #############################
        # Action & Observation Spaces

        # Action space
        self.action_shape = [ self.num_agents ]              # an action for every agent
        #--------------------------------------------------------
        # self.action_space = spaces.Box(
        #     low=0, high=self.num_grids, shape=self.action_shape, dtype=int)   # 0 to num_grids: transit to the grid with that index in self.grids; if grid == self.locs[agent], then it means stay and serve
        #--------------------------------------------------------
        # self.action_space = spaces.MultiDiscrete([
        #     self.num_grids+1 for i in range( self.num_agents )
        # ])
        #--------------------------------------------------------
        # [x] Does not work...
        # self.action_space = spaces.MultiDiscrete([
        #     [ 3, self.num_grids+1 ] for i in range( self.num_agents )
        # ])   # The first one denotes the mode: 0 means idle, 1 means x, 2 means y
        #--------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(np.concatenate([
            [ 3, self.num_grids ] for i in range( self.num_agents )
        ]))   # The first one denotes the mode: 0 means idle, 1 means x, 2 means y


        # Observation space
        ### -> Only demands
        dmd_observation_shape = [ self.num_grids, self.params['pred_grid'], 2 ]   # only predicted demands, with mean and variance...
        self.observation_space = spaces.Dict( #spaces.Box(low=0, high=data['internet'].max(), shape=dmd_observation_shape, dtype=np.float16),   # Demand observation space
            {
                'preds': spaces.Box(low=0, high=self.data['internet'].max(), shape=dmd_observation_shape, dtype=np.float16),   # Demand observation space
                'locs': spaces.Box(low=0, high=len(self.grids)-1, shape=[ self.num_agents ], dtype=int),                         # Location observation space
            })


        # Reset seeds
        random.seed(params['seed'])
        np.random.seed(params['seed'])

        # Output
        print("Environment initialized.")



    def _next_observation(self):
        '''Generate the next observation.'''
        self.obs = {
                'preds': self.preds[ self.ti ][:, 0:self.params['pred_grid'], :],
            'locs': self.locs,
        }
        return self.obs



    def reset(self, preds = None, gridmap = None):
        tt = time.time()
        # Reset seeds
        random.seed(params['seed'])
        np.random.seed(params['seed'])

        # Reset simulation snapshot variables
        ###############-> ti is reset to start_ti instead of beginning of dataset
        self.ti = self.start_ti      #params['predict_input_size']     # current time slot index to MAKE A PREDICTION/DECISION, index of self.times
        self.locs = [ 0 for i in range(self.num_agents) ]    # current location of each vehicle
        self.reward = 0
        self.done = False

        # Reset result counters
        self.rewards = []
        self.grid_rewards = []
        self.grid_utils = []
        self.grid_costs = []

        # # Start initial predictions
        # features, labels = make_prediction_seqs( self.data, self.ti )

        # For simplicity of the environment, we make the prediction all at once, at the beginning of each game/simulation
        if preds is None or gridmap is None:
            if self.preds is None or self.gridmap is None:
                print("Environment resetting. All-grid prediction in progress...")
                #self.preds, self.gridmap = predict_all( self.data, self.model, self.params )
            else:
                print("Predictions already in-place. Skipping predictions.")
        else:
            print("Prediction passed as input.")
            self.preds = preds
            self.gridmap = gridmap

        # Obtain the first observation
        self._next_observation()

        # Return the observation
        print("Environment reset finished. Time: %.2f s" % (time.time()-tt))
        return self.obs




    ###################################################################################
    # Experimentation
    ###################################################################################
    '''
    def solve_cvar_sdp(self, mu, std, z, grid_type = 0):
        #Solve the worst-case CVaR SDP to calculate the cost (excess of resources) of a cell.
        eps = self.params['eps']
        
        # Variables
        M = cp.Variable((2, 2), PSD=True)  # PSD=True implies symmetric; if not PSD=True, then we can set symmetric=True
        nu = [ cp.Variable((2, 2)) for k in range(params['num_resources']) ]
        beta = cp.Variable((1, 1))
        w = cp.Variable((1, params['num_resources']))
        x = cp.Variable((1, 1))
        Omega = np.array( [ [mu**2 + std**2, mu], [mu, 1] ] )
        
        # Constraint 2
        constraints = []
        for k in range(params['num_resources']):
            constraints += [ M - nu[k] >> 0 ]
            constraints += [ nu[k][0,0] == 0 ]
            constraints += [ nu[k][0, 1] == 1/2 * params['f_coeffs'][grid_type][k] / params['g'][k] ]
            constraints += [ nu[k][1, 0] == 1/2 * params['f_coeffs'][grid_type][k] / params['g'][k] ]
            constraints += [ beta == (params['f_consts'][grid_type][k] - params['r_consts'][k]) / params['g'][k] - nu[k][1, 1] - z - w[0, k] ]
            constraints += [w[0, k] >= 0]
            if params['f_coeffs'][grid_type][k] > 1e-8:
                constraints += [ x >= w[0, k] / params['f_coeffs'][grid_type][k] ]
                
        constraints += [ cp.trace(M @ Omega) / eps + beta <= 0 ]
        
        # Objective
        obj = x
        
        # Problem
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        # Solve the problem
        try:
            # val = prob.solve(solver=cp.CVXOPT, verbose=True)
            obj = prob.solve(solver=cp.MOSEK)#, verbose=True)
        except:
            print("mu, std, z:", mu, std, z)
            raise Exception("Wrong with CVXPY solving")
        
        return obj   # Return the amount of excessive resources needed for each resource type; numerical errors addressed by explicitly judging for 0
        # return val
       ''' 


    def _get_reward(self, action, locs = None, transit = True):
        '''
        Take an action for the current time. This involves solving the CVaR SDP, deciding costs, and calculating rewards.
        - transit: whether to actually change the state of the environment, or just take the action and calculate the reward.
        '''
        ### When using Tuple( [ Discrete() ] ) action space, this rounding is not needed.
        #action = [ int(round(a)) for a in action ]
        if locs is None:
            locs = self.locs

        # 1. Calculate resources (and costs) per grid, and take the action
        zat = [ 0 for grid in self.grids ]
        costs = list(np.zeros(self.num_agents))
        for i in range(self.num_agents):
            if action[2*i] == 0:                      # IDLE
                costs[i] = params['i_cost']
            elif action[2*i] == 1:                    # STAY-AND-SERVE
                if self.nbrs is None:
                    zat[ locs[i] ] += 1
                else:
                    grid = self.grids[ action[2*i+1] ]
                    # Enumerate all grids
                    for i, g in enumerate(self.grids):
                        if g in self.nbrs[grid]:
                            zat[i] += 1
                costs[i] = params['x_cost']
            elif action[2*i] == 2:                    # IN-TRANSIT
                locs[i] = action[2*i+1]
                costs[i] = params['y_cost']
        # 2. Calculate rewards
        penalty = [ 0 for grid in self.grids ]
        util = [ 0 for grid in self.grids ]
        ress = [ None for grid in self.grids ]
        leftover = [ None for grid in self.grids ]
    
        for i, grid in enumerate(self.grids):
        
            grid_type = 0
            if self.grid_types is not None:
                grid_type = self.grid_types[grid]
        
            dmd = self.demands[i][ self.ti ]
            f = self.params['f_coeffs'][grid_type] * dmd + self.params['f_consts'][grid_type]
            ress[i] = self.params['r_consts'] + self.params['g'] * zat[i]
            ''' 
            print('-------------')
            print(grid_type)
            print(dmd)
            print(f, sum(f))
            print(ress[i], sum(ress[i]))
            print(ress[i]-f)
            '''
            if (np.array(ress[i] - f) > 0).all():
                util[i] = -np.max(self.params['u_coeffs'] / (ress[i] - f))
                util[i] = np.max([self.params['base_util'], util[i]])
                leftover[i], penalty[i] = 0, 0
            else:
                util[i] = self.params['base_util']
                for k in range(params['num_resources']):
                    if params['f_coeffs'][grid_type][k] > 1e-8:
                        leftover[i] = np.max((f - ress[i])/self.params['f_coeffs'] )
                        penalty[i] =  leftover[i] * self.params['penalty_w']
        '''
        print('util:', util)
        print('penalty:', penalty)
        print('cost:', costs)
        '''
        reward  = np.sum(util) - np.sum(penalty) - np.sum(costs)

        #----------------------------
        # Take the actual transit in the environment
        if transit:
            # Action
            self.last_action = action

            # Util/Penalty/Cost
            self.reward = reward
            self.util = util
            self.costs = costs
            self.penalty = penalty
            self.leftover = leftover

            # Locations
            self.locs = locs
        #----------------------------

        return reward
 

    def dp_stay(self, ti, t_s, locs, previous):
        
        grid_type = 0
        if self.grid_types is not None:
            grid_type = self.grid_types[self.grids[locs]]
        
        mu = self.preds[ti][ locs , t_s, 0]
        std = self.preds[ti][ locs , t_s, 1]
    
        f = self.params['f_coeffs'][grid_type] * mu + self.params['f_consts'][grid_type]
        ress = self.params['r_consts'] + self.params['g'] * previous
        ress_prev = self.params['r_consts'] + self.params['g'] * (previous-1)
    
        # utility
        util = -np.max( self.params['u_coeffs'] / (ress - f))  
        if util >= 0: 
            util = params['base_util']
        util = np.max([util, params['base_util']])
    
        util_prev = -np.max(self.params['u_coeffs'] / (ress_prev - f))
        if util_prev >= 0: 
            util_prev = params['base_util']
        util_prev = np.max([util_prev, params['base_util']])

        # penalty
        #leftover = self.solve_cvar_sdp( mu, std, previous, grid_type)
        #leftover_prev = self.solve_cvar_sdp( mu, std, (previous-1), grid_type)
        msz = torch.FloatTensor([mu, std, previous])
        msz_prev = torch.FloatTensor([mu, std, previous-1])
        leftover = 0.01 * self.model(msz.cuda()).cpu().detach().numpy() 
        leftover_prev = 0.01 * self.model(msz_prev.cuda()).cpu().detach().numpy()  
        
        # when leftover < 0 (ress > f), leftover = 0
        if leftover < -1e-8 or leftover_prev < -1e-8:
            print('Wrong for leftover!!!!!!!!!!!!!!!!!!')
        
        penalty = leftover * self.params['penalty_w']
        penalty_prev = leftover_prev * self.params['penalty_w']
        
        score = (util - util_prev) - (penalty - penalty_prev) - params['x_cost']
        '''
        print('dp_stay--------------------------')
        print('dp time:', ti)
        print('loc:', locs)
        print('prev:', previous)
        print('util:', util, 'util_prev:', util_prev)
        print('penalty:', leftover, penalty, 'penalty_prev:',leftover_prev,  penalty_prev)
        '''
        return score


    def dp_move(self):
        return -params['y_cost']

    def dp_idle(self):
        return -params['i_cost']



    def step(self, action):
        '''Take a time step over the current state, including taking the action, and observing environmental changes'''
        #-----------------------------------------------------
        # Take baseline action if the result is not saved...
        if params['use_baseline']:
            if self.baseline_rewards[self.ti] is None:
                self.baseline_rewards[self.ti] = self._take_action( self.baseline_action, locs=self.baseline_locs, transit=False )

        # Take actual action
        self._get_reward(action)          # Compute self.reward


        #-----------------------------------------------------
        # Subtract baseline reward
        if params['use_baseline']:
            self.reward -= self.baseline_rewards[self.ti]


        ######
        # Train output
        ######
        if params['train_output']:
            print("Step", self.ti, 'reward:', self.reward)


        # Advance time
        self.ti += 1                       # Advance time
        ###############-> condition for "done" is reaching end_ti, not the end of dataset
        if (self.ti >= self.end_ti):    #len(self.times)):   # Check if done
            self.obs = None
            self.done = True
        else:
            self._next_observation()           # Update self.obs
        return self.obs, self.reward, self.done, {}
