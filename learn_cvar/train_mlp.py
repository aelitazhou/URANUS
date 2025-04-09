import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model import MLP
import random

seed = 2014
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# data loading
n_a = ['_1', '_2', '_3']
n_g = ['2', '3', '6', '12']
scalar = 200
cvar_x = []
cvar_y = []
for i in n_a:
    for j in n_g:
        feature = np.load('cvar_data/last_0.8/cvar_x_' + j + i + '.npy')
        label = np.load('cvar_data/last_0.8/cvar_y_' + j + i + '.npy')
        cvar_x.append(feature)
        cvar_y.append(label)
        print(feature.shape, label.shape)
cvar_x = np.array(cvar_x).reshape((-1, 3))
cvar_y = scalar * np.array(cvar_y).reshape((-1, 1))
print('Train X shape:', cvar_x.shape)
print('Train_Y shape:', cvar_y.shape)
np.save('cvar_data/last/cvar_x', cvar_x)
np.save('cvar_data/last/cvar_' + str(scalar) + 'y', cvar_y)
print('------------Training-------------')

n_epochs = 500
batch_size = 640
learning_rate = 0.001
momentum = 0.9

name = 'mlp' + str(scalar) + '_' + str(seed) 

def train():
    ds_train = torch.utils.data.TensorDataset(torch.FloatTensor(cvar_x), torch.FloatTensor(cvar_y))
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    model = MLP().cuda()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)
    lossfunc = torch.nn.MSELoss()
    #lossfunc = torch.nn.L1Loss()

    for epoch in range(n_epochs):
        for i, (feature, label) in enumerate(dataloader_train):
            optimizer.zero_grad()
            output = model(feature.cuda())
            loss = lossfunc(output, label.cuda())
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss,  flush=True)
    torch.save(model.state_dict(), 'Lcvar_' + name + '.model')

if __name__ == "__main__":
    train()
