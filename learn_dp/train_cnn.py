import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model import CNN
np.random.seed(2022)

# data loading
n_g = '9'

# 2, 0.0005; 3, 0.00025; 6, 0.0001
#scalar = 0.0005

dps_x = np.load('ldp_train_data/dps_x_' + n_g + '_3.npy')
dps_y = np.load('ldp_train_data/dps_y_' + n_g + '_3.npy')
#dps_y = scalar * dps_y

print('Train X shape:', dps_x.shape)
print('Train_Y shape:', dps_y.shape)
np.save('data_npy/dps_x_' + n_g, dps_x)
#np.save('data_npy/dps_' + str(scalar) + 'y_' + n_g, dps_y)
np.save('data_npy/dps_y_' + n_g, dps_y)
print('------------Training-------------')

n_epochs = 100
batch_size = 640
learning_rate = 0.00001
momentum = 0.9
kernel_size = 1 

#name = 'mlp' + str(scalar) + '_' + str(n_g)
name = 'cnn' + str(n_g)
print(name)
train_x = dps_x.reshape(dps_x.shape[0], 1, dps_x.shape[1], dps_x.shape[2])

def train(train_x, train_y, k_size):
    ds_train = torch.utils.data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    model = CNN(train_x.shape[2], train_x.shape[3], k_size).cuda()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)
    lossfunc = torch.nn.MSELoss()
    #lossfunc = torch.nn.L1Loss()

    for epoch in range(n_epochs):
        t = time.time()
        for i, (feature, label) in enumerate(dataloader_train):
            optimizer.zero_grad()
            output = model(feature.cuda())
            loss = lossfunc(output, label.cuda())
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss, 'time:', time.time()-t, flush=True)
    torch.save(model.state_dict(), 'Ldp_optimal_' + name + '.model')


#if __name__ == "__main__":
 #   train(train_x, dps_y, kernel_size)

