import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error,  mean_squared_error

# data loading
# data loading
n_a = ['_1', '_2', '_3']
n_g = ['1', '2', '3', '6', '9', '12']
scalar = 100
train_len = 3456 # 80%
cvar_x = []
cvar_y = []
for i in n_a:
    for j in n_g:
        cvar_x.append(np.load('cvar_x_' + j + i + '.npy')[train_len:, :])
        cvar_y.append(np.load('cvar_y_' + j + i + '.npy')[train_len:])
cvar_x = np.array(cvar_x).reshape((-1, 3))
cvar_y = scalar * np.array(cvar_y).reshape((-1, 1))
print('Test X shape:', cvar_x.shape)
print('Test_Y shape:', cvar_y.shape)
print('------------Testing-------------')


class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP,self).__init__()
    self.fc1 = torch.nn.Linear(3,64)
    self.fc2 = torch.nn.Linear(64,32)
    self.fc3 = torch.nn.Linear(32, 1)

  def forward(self, din):
    dout = F.relu(self.fc1(din))
    dout = F.relu(self.fc2(dout))
    dout = F.relu(self.fc3(dout))
    return dout


def test():
    ds_test = torch.utils.data.TensorDataset(torch.FloatTensor(cvar_x), torch.FloatTensor(cvar_y))
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=864, shuffle=False)

    model = MLP().cuda()
    model.load_state_dict(torch.load('lc.model'))
    model.eval()

    pred = []
    for i, (feature, label) in enumerate(dataloader_test):
        output = model(feature.cuda()).detach().cpu().numpy()
        pred.append(output)
    pred = np.array(pred).reshape(-1, 1)
    print('MSE:', mean_squared_error(cvar_y, pred))
    #print('MAE:', mean_absolute_error(cvar_y, pred))
    np.save('true', cvar_y)
    np.save('pred', pred)



if __name__ == "__main__":
    test()
