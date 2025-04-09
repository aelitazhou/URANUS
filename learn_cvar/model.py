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


class Actor_Critic(nn.Module):
    def __init__(self, inputs, outputs, outputs_, hidden_size, std=0.0):
        super(Actor_Critic, self).__init__()
        self.Actor = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outputs),
            nn.Softmax(dim=1))
        self.Actor_ = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outputs_),
            nn.Softmax(dim=1))
        self.Critic = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1))
    def forward(self,x):
        value = self.Critic(x)
        probs = self.Actor(x)
        probs_ = self.Actor_(x)
        dist = Categorical(probs)
        dist_ = Categorical(probs_)
        return probs, probs_, dist, dist_, value


class Share_layer(nn.Module):
    def __init__(self, inputs, hidden_size): 
        super(Share_layer, self).__init__()
        self.linear1 = nn.Linear(inputs, hidden_size)
    def forward(self, out):
        out = self.linear1(out)
        out = F.relu(out)
        return out
        
        
class Actor(nn.Module):
    def __init__(self, sl, hidden_size, outputs):
        super(Actor, self).__init__()
        self.share_layer = sl
        self.linear2 = nn.Linear(hidden_size, outputs)
    def forward(self, out):
        out = self.share_layer(out) 
        out = self.linear2(out)
        prob = F.softmax(out, dim = 1)
        return prob, out


class Actor_(nn.Module):
    def __init__(self, sl, hidden_size, outputs):
        super(Actor_, self).__init__()
        self.share_layer = sl
        self.linear2 = nn.Linear(hidden_size, outputs)
    def forward(self, out):
        out = self.share_layer(out)
        out = self.linear2(out)
        prob = F.softmax(out, dim = 1)
        return prob, out

    
class Critic(nn.Module):
    def __init__(self, sl, hidden_size):
        super(Critic, self).__init__()
        self.share_layer = sl
        self.linear2 = nn.Linear(hidden_size, 1)
    def forward(self, out):
        out = self.share_layer(out)
        out = self.linear2(out)
        return out
