import os, sys, time, random, json, pickle, itertools
import numpy as np
import pandas as pd
import networkx as nx

import torch, torchvision
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from model import RLSTM, BLSTM, BLSTM_NMC, LSTM_MC, LSTM_NMC

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


print('Current CUDA device:', torch.cuda.get_device_name(0))
torch.cuda.set_device(0)
t = time.time()
data = 6000*np.load('demands.npy')[0]
print('Data shape:', data.shape)

# Train/Test data
#x = datax['internet'].to_numpy()
total_size = data.shape[0]
train_size = 144*15
test_size = total_size - train_size
print("Train size:", train_size, "Test size:", test_size)

s_range = [0, 1]
scaler = MinMaxScaler(feature_range=(s_range[0], s_range[1]))
data_n = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
train = data_n[:train_size]
test = data_n[train_size:]
print('Train data shape:', train.shape)

def create_in_data(input_data, tw_l, cuda=False):
    in_data = []
    L = len(input_data)
    for i in range(L-tw_l):
        seq = input_data[i:i+tw_l+1]
        in_data.append(seq)
    return np.array(in_data).reshape(-1, tw_l+1)
tw = 144
step = 12
tw_l = tw + step -1
train_seq = np.array(create_in_data(train, tw_l))
train_seq = torch.FloatTensor(train_seq)
print('Train seq shape:', train_seq.shape)
print('------------Training------------')

# Training

# paras
epochs = 400
batch_size = 128
# training h_para
# 0.001 for Lstm, Lstm_mc, Lstm_nmc; 0.0001 for Blstm, Blstm_nmc
l_rate = 0.001
w_decay = 0.0005

# dataloader
ds_train = torch.utils.data.TensorDataset(train_seq[:, 0:tw], train_seq[:, tw:])
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

# Models

# BLSTM
name = 'Lstm_'
model = RLSTM(1, 50, 1, step).cuda()

# optim and metric
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=w_decay)
criterion = torch.nn.MSELoss()

# Start training
model.train()
ttt = time.time()
#prev_loss = torch.ones(train_seq.shape[0]).cuda()
for i in range(epochs):
    tt = time.time()
    for j, (seq, label) in enumerate(dataloader_train):
        optimizer.zero_grad()
        
        seq = seq.cuda()
        label = label.cuda()
                
        output = model(seq)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        
    print(f'epoch: {i:3} loss: {loss.item():10.10f}\t time: {time.time()-tt}', flush=True)

print(f'Total training time: {time.time()-ttt}')

torch.save(model.state_dict(), 'trained_models/' + name + '.model')
