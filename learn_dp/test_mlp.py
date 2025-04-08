import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error,  mean_squared_error
from model import MLP

# data loading
# data loading
n_a = ['_1', '_2', '_3']
n_g = ['1', '2', '3', '6', '9', '12']
scalar = 200

cvar_x = []
cvar_y = []
for i in n_a:
    for j in n_g:
        feature = np.load('lcvar_test_data/cvar_x_' + j + i + '.npy')
        ts = int(j) * np.ones(feature.shape[0])
        feature = np.column_stack((feature, ts))
        label = np.load('lcvar_test_data/cvar_y_' + j + i + '.npy')
        cvar_x.append(feature)
        cvar_y.append(label)
cvar_x = np.array(cvar_x).reshape((-1, 4))
cvar_y = scalar * np.array(cvar_y).reshape((-1, 1))
print('Test X shape:', cvar_x.shape)
print('Test_Y shape:', cvar_y.shape)
print('------------Testing-------------')


name = 'mlp'
def test():
    ds_test = torch.utils.data.TensorDataset(torch.FloatTensor(cvar_x), torch.FloatTensor(cvar_y))
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
