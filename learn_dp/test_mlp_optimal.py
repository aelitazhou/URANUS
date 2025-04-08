import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error,  mean_squared_error
from model import MLP
np.random.seed(2022)

# data loading
n_g = '2'
# 2, 0.0005; 3, 0.00025; 6, 0.0001
scalar = 0.0005

dps_x = np.load('ldp_train_data/dps_x_' + n_g + '_3.npy').reshape(-1, 4)[:, 0:3]
dps_y = np.load('ldp_train_data/dps_y_' + n_g + '_3.npy').reshape(-1, 1)
dps_y = scalar * dps_y

idx = np.where(dps_x[:, -1] != 0)[0].reshape(-1, 1)
dps_x = dps_x[idx, :].reshape(-1, 3)
dps_y = dps_y[idx, :].reshape(-1, 1)

print('Test X shape:', dps_x.shape)
print('Test_Y shape:', dps_y.shape)


name = 'mlp' + str(scalar) + '_' + str(n_g)
print(name)

print('------------Testing-------------')


def test():
    ds_test = torch.utils.data.TensorDataset(torch.FloatTensor(dps_x), torch.FloatTensor(dps_y))
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=int(cvar_y.shape[0]/18), shuffle=False)

    model = MLP().cuda()
    model.load_state_dict(torch.load('Lcvar_' + name + str(scalar) + '.model'))
    model.eval()

    pred = []
    for i, (feature, label) in enumerate(dataloader_test):
        output = model(feature.cuda()).detach().cpu().numpy()
        pred.append(output)
    pred = np.array(pred).reshape(-1, 1)
    print('MSE:', mean_squared_error(cvar_y, pred))
    #print('MAE:', mean_absolute_error(cvar_y, pred))
    np.save('true' + str(scalar), cvar_y)
    np.save('pred' + name + str(scalar), pred)



if __name__ == "__main__":
    test()
