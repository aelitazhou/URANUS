import torch, torchvision
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical
import torch.nn.functional as F
from blitz.modules import BayesianLinear, BayesianLSTM
from blitz.utils import variational_estimator
torch.manual_seed(2022)


class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP,self).__init__()
    self.fc1 = torch.nn.Linear(3,256)
    self.fc2 = torch.nn.Linear(256,128)
    self.fc3 = torch.nn.Linear(128, 64)
    self.fc4 = torch.nn.Linear(64, 32)
    self.fc5 = torch.nn.Linear(32, 1)
    #self.fc6 = torch.nn.Linear(16, 1)

  def forward(self, din):
    dout = F.relu(self.fc1(din))
    dout = F.relu(self.fc2(dout))
    dout = F.relu(self.fc3(dout))
    dout = F.relu(self.fc4(dout))
    dout = F.relu(self.fc5(dout))
    #dout = F.relu(self.fc6(dout))
    return dout


class MLP_F(torch.nn.Module):
  def __init__(self, input_size):
    super(MLP_F,self).__init__()
    self.fc1 = torch.nn.Linear(input_size, 256)
    self.fc2 = torch.nn.Linear(256,128)
    self.fc3 = torch.nn.Linear(128, 64)
    self.fc4 = torch.nn.Linear(64, 32)
    self.fc5 = torch.nn.Linear(32, 10)

  def forward(self, din):
    dout = F.relu(self.fc1(din))
    dout = F.relu(self.fc2(dout))
    dout = F.relu(self.fc3(dout))
    dout = F.relu(self.fc4(dout))
    dout = F.relu(self.fc5(dout))
    return dout


class CNN(nn.Module):
    def __init__(self, input_h, input_w, k_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, k_size) # in_channel, out_channel, kernel size
        self.conv2 = nn.Conv2d(3, 12, k_size)
        self.fc1 = nn.Linear(12 * (input_h-2*(k_size-1)) * (input_w-2*(k_size-1)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_3(nn.Module):
    def __init__(self, input_h, input_w, k_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, k_size) # in_channel, out_channel, kernel size
        self.conv2 = nn.Conv2d(3, 12, k_size)
        self.fc1 = nn.Linear(12 * (input_h-2*(k_size-1)) * input_w, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class LSTM_MC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_MC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, mask_in, mask_out):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).cuda()
        # forward propagate lstm
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        mask_in = mask_in.reshape(x.shape)
        x = x*mask_in

        out, (h_n, h_c) = self.lstm(x, (h0, c0)) 
        mask_out = mask_out.reshape(out.shape)
        out = out*mask_out

        out = self.fc(out[:, -1, :])
        return out


class LSTM_NMC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_rate):
        super(LSTM_NMC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.drop_rate = drop_rate

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).cuda()

        # forward propagate lstmx = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        x = F.dropout(x, p=self.drop_rate, training=True)
        
        out, (h_n, h_c) = self.lstm(x, (h0, c0))
        out = F.dropout(out, p=self.drop_rate, training=True)
        
        out = self.fc(out[:, -1, :])
        return out


class RLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # initialization
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).cuda()

        # forward propagate lstm
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        out, (h_n, h_c) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



@variational_estimator
class BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(BLSTM, self).__init__()
        '''
        self.lstm_1 = BayesianLSTM(input_dim, hidden_size).cuda()
        self.linear = nn.Linear(hidden_size, output_dim).cuda()
        '''
        self.lstm_1 = BayesianLSTM(input_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        x_, _ = self.lstm_1(x)
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_


@variational_estimator
class BLSTM_NMC(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, drop_rate):
        super(BLSTM_NMC, self).__init__()
        self.lstm_1 = BayesianLSTM(input_dim, hidden_size).cuda()
        self.linear = nn.Linear(hidden_size, output_dim).cuda()
        self.drop_rate = drop_rate

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        x = F.dropout(x, p=self.drop_rate, training=True)

        x_, _ = self.lstm_1(x)
        x_ = F.dropout(x_, p=self.drop_rate, training=True)

        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_



class ActorCritic(nn.Module):
    def __init__(self, inputs, outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        self.Actor = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outputs),
            nn.Softmax(dim=1))
        self.Critic = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1))
    def forward(self,x):
        value = self.Critic(x)
        probs = self.Actor(x)
        dist = Categorical(probs)
        return probs, dist, value
