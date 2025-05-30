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

    def __init__(self, params, data, dmds, grid_map, nbrs = None, sorted_grids = None, preds = None, gridmap = None):
        '''[TODO]: Some grids have missing timestamps possibly due to 0 activity. Should make up for those and fill the 0 values. For now we just use the top 50 grids which do not have this issue.'''
        super(MOESimulator, self).__init__()

        # Reset seeds
        random.seed(params['seed'])
        np.random.seed(params['seed'])

        # Initialize parameters
        self.params = params
        self.num_agents = params['num_agents']
        self.num_grids = params['num_grids']
        self.grid_map = grid_map.copy() # grid map
        self.grid_mp = np.empty((params['num_agents'], params['num_grids'], params['num_grids'])) # time slots distance for moving process
        for i in range(params['num_agents']):
            self.grid_mp[i] = grid_map.copy()


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
            [ self.num_grids ] for i in range( self.num_agents )
        ]))   # only output locations


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
        self.r_locs = [ 0 for i in range(self.num_agents) ]    # current location of each vehicle for calculating reward
        self.reward = 0
        self.done = False
        self.des_locs = self.locs.copy()
        self.sor_locs = self.locs.copy()

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


    def _get_reward(self, action, locs = None, transit = True):
        '''
        Take an action for the current time. This involves solving the CVaR SDP, deciding costs, and calculating rewards.
        - transit: whether to actually change the state of the environment, or just take the action and calculate the reward.
        '''
        ### When using Tuple( [ Discrete() ] ) action space, this rounding is not needed.
        #action = [ int(round(a)) for a in action ]
        if locs is None:
            locs = self.r_locs 
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
                #locs[i] = action[2*i+1]
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

        #----------------------------

        return reward


    def reloc(self, action_):
        # action rebuild
        action = []
        for i, j in enumerate(self.r_locs):
            if j == 100:
                action.append(2)
                action.append(100)
            else:
                if action_[i] == j:
                    action.append(1)
                    action.append(action_[i])
                else:
                    action.append(2)
                    action.append(action_[i])

        # follow map
        action_new = action.copy()
        new_locs = self.r_locs.copy()
        for a in range(self.num_agents):
            des = int(action[2*a+1])
            sor = int(self.r_locs[a])
            if self.r_locs[a] != 100:
                self.des_locs[a] = des
                self.sor_locs[a] = sor
            if action[2*a] == 2:
                if action[2*a+1] != 100: # decision made
                    # reachable in one time slot distance
                    if self.grid_mp[a][sor][des] == 1:
                        action_new[2*a+1] = des
                        new_locs[a] = des
                        # not reachable in one time slot distance
                    else:
                        self.grid_mp[a][sor][des] -= 1
                        action_new[2*a+1] = 100
                        new_locs[a] = 100
                if action[2*a+1] == 100:  # in the moving process, action state = 2
                    sor = self.sor_locs[a]
                    des = self.des_locs[a]
                    if self.grid_mp[a][sor][des] > 1:
                        self.grid_mp[a][sor][des] -= 1
                        action_new[2*a+1] = 100
                        new_locs[a] = 100
                    if self.grid_mp[a][sor][des] == 1:
                        self.grid_mp[a][sor][des] = self.grid_map[sor][des]
                        action_new[2*a+1] = des
                        new_locs[a] = des
        # location update
        for a in range(self.num_agents):
            self.r_locs[a] = new_locs[a]
            if new_locs[a] != 100:
                self.locs[a] =new_locs[a]
        print('sor:', self.sor_locs)
        print('des:', self.des_locs)
        print('r_loc:', self.r_locs)
        print('loc:', self.locs)
        return action_new


    def step(self, action):
        '''Take a time step over the current state, including taking the action, and observing environmental changes'''
        #-----------------------------------------------------
        # Take baseline action if the result is not saved...
        if params['use_baseline']:
            if self.baseline_rewards[self.ti] is None:
                self.baseline_rewards[self.ti] = self._take_action( self.baseline_action, locs=self.baseline_locs, transit=False )

        # Take actual action
        self._get_reward(self.reloc(action))          # Compute self.reward


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
