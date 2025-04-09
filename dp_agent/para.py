import numpy as np

params = {
    'seed': 12345678, # 12345678
    'seed_rng': [12345678, 2837018475, 275708450, 3051424498, 3069252051, 1744240523, 2799297416, 
                  1478761707, 558578666, 3280475235, 1894759115, 3228592332, 692862313, 183869678, 
                  1756858403, 617664758, 2773734463, 3933278132, 2138323871, 2034216651, 4209663376, 
                  3882207967, 2950005990, 3432484138, 3670643452, 811134793, 3298449689, 3703583643, 
                  2229503679, 1191140425, 269390302, 2568457748, 2768325726, 4161890726, 1487968251, 
                  3126294104, 1778255325, 3281478366, 4191879050, 1750480184, 1831762826, 163473697, 
                  3934602837, 888470314, 2616538842, 1133428815, 3052303722, 3911671254, 1019817809, 
                  2157694103, 2755067755, 2678358978, 3398794671, 2891691292, 1423654999, 3914102306, 
                  560310532, 3902888695, 1246699533, 449750433, 295271363, 3761722022, 1318113951, 
                  358594136, 3880953492, 21325223, 3629680856, 4013771052, 958176799, 1263925355, 
                  4135841266, 998433578, 1121682120, 683027317, 2305274283, 277103835, 1793776223, 
                  711443840, 658992155, 3564406101, 1532248546, 4167142763, 1361120171, 409489366, 
                  2598279910, 2299048564, 2699743697, 1536054717, 4048678376, 2790280749, 447875870, 
                  201259587, 3330325009, 2224141540, 1579587490, 2287421573, 3018374907, 1605560336, 
                  2597273107, 1228959680],        # 1 default seed + 99 truly random seeds, generated at https://www.gigacalculator.com/calculators/random-number-generator.php
    
    # Parameters related to the time series
    #'num_agents': 3,
    'num_grids': 10,               # scheduling with top 50 grids
    'predict_input_size': 144,     # use 144 slots in the past to predict
    'predict_output_size': 12,     # predict the next 12 slots (2 hours)
    'predict_model': 'Blstm12_10g',    # BNN prediction model
    'bnn_samples': 100,            # number of samples to estimate mean & variance
    
    'demand_scaling_ratio': 6000,  # How much to scale the demands such that they are on the same order as the CPU/GPU/memory   => For numerical stability of SDP 
    
    'neighbor_distance_threshold': 600,   # Threshold of distance for two grids to be neighbors
    
    # Resource allocation parameters
    'num_resources': 3,            # CPU, GPU, memory
    'f_coeffs': [  # The two dimensions are about the "type" of a grid: whether it is an on-grid (demand higher between 6am-6pm), or if it is an off-grid (demand higher between 6pm-6am)
        np.array([ 0, 0, 30 ]),   # CPU/GPU are constants (coeff=0), memory is linear (coeff=0.001 GB/demand -- 10GB/6000 people)        # \phi
        np.array([ 0, 0, 10 ]),   # CPU/GPU are constants (coeff=0), memory is linear (coeff=0.001 GB/demand -- 10GB/6000 people)        # \phi
    ],
    # memory goes up as dmd goes up
    # memory only affect wss, not utility
    'f_consts': [
        np.array([ 8, 1, 8 ]),       # CPU: at least 8 cores for instantiation; GPU: at least 1 for instantiation; memory: 1GB at least   # \varphi
        np.array([ 8, 1, 8 ]),       # CPU: at least 8 cores for instantiation; GPU: at least 1 for instantiation; memory: 1GB at least   # \varphi
    ],
    'r_consts': np.array([ 16, 2, 64 ]),     # Static resources at each cell; assumed to be uniform in all cells                                  # r
    'g':        np.array([ 16, 2, 32 ]),     # 16 cores, 2 GPUs, 32GB; per MU
    
    'eps':      0.05,                        # probability defining the CVaR
    'penalty_w':   500,                      # penalty weight, large scalar
    
    # Utility functions
    # 1. Queueing delay utility
    'u_coeffs': np.array([ 800, 100, 0 ]),       # utility function shape = - max_k {  dmd * u_coeffs[k] / resource  }     => Bottleneck queueing delay of different resources
    # secondary cost
    # 2. CVaR constraint - excessive resource costs
    'w_coeffs': np.array([ 10000, 10000, 10000 ]),              # how much does it cost for any resource
    # enlarge violation cast, primary goal
    # 3. No utility?
    'no_util': False,                       # Do we contain utility (u) in the reward?
    'no_cvar': False,                       # Do we contain CVaR penalty (w) in the reward?
    'no_cost': False,                       # Do we contain x/y/i costs in the reward?
    
    # 4. Additional cost factors
    'x_cost': 50,                           # Stay-and-serve cost per time slot(10)
    'y_cost': 10,                          # Moving cost per time slot(100)
    'i_cost': 0,                            # Idle cost per time slot
    'discount_factor': 0.8,                

    # reward normalization/substract baseline reward(static top five grids reward), aiming to stablize the training process
   
    'base_util':   -120,                      # base line utility, -100/-33.3
    'pred_grid': 6,
    'num_agents': 1,
    'model_cvar': 'model/map/3f_last_0.8/Lcvar_mlp200_2014.model',   # 2028, 2027, 2026, 2024, 2022
    'scalar_back': 0.005,    # 0.01 for 100; 0.02 for 50

    # DRL parameters
    # Use baseline: add a baseline to RL such that
    'use_baseline': False,
    
    
    ##############################
    # Debug params
    'debug_output': False,
    'train_output': True,

    # learning rate
    'lr': 0.0001,     
}
