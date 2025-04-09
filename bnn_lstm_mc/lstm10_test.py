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
data = 6000*np.load('demands.npy')
print('Data shape:', data.shape)

# Train/Test data
#x = datax['internet'].to_numpy()
total_size = data.shape[1]
train_size = 144*15
test_size = total_size - train_size
print("Train size:", train_size, "Test size:", test_size)

scaler = MinMaxScaler(feature_range=(-1, 1))
data_n = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
train = data_n[:, :train_size]
test = data_n[:, train_size:]
print('Test data shape:', test.shape)

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
test_seq = np.array(create_in_data(test[0], tw_l))
for i in range(test.shape[0]-1):
    test_seq = np.vstack((test_seq, create_in_data(test[i+1], tw_l)))
test_seq = torch.FloatTensor(test_seq)
print('Test seq shape:', test_seq.shape)
print('------------Testing------------')

# Testing
name = 'Lstm10'
model = RLSTM(1, 50, 1, step).cuda()

model.load_state_dict(torch.load('trained_models/' + name + '.model'))
model.eval()

# dataloader
batch_size = int(test_seq.shape[0]/10)
ds_test = torch.utils.data.TensorDataset(test_seq[:, 0:tw], test_seq[:, tw:])
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

t_step = int(test_seq.shape[0]/10)
preds = np.empty([t_step, 10, step])
pred = np.empty([test_seq.shape[0], step])
for j, (seq, label) in enumerate(dataloader_test):
    p = model(seq.cuda()).cpu().detach().numpy()
    pred[batch_size*j:batch_size+batch_size*j, :] = p

pred = np.array(scaler.inverse_transform(pred.reshape(-1, 1))).reshape(pred.shape)
for j in range(10):
    loc_pred = pred[j*t_step:(j+1)*t_step, :]
    preds[:, j, :] = loc_pred 

print('Preds shape:', preds.shape)

np.save('test_results/' + name + '_pred', preds)
