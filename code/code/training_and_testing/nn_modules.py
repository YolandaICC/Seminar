import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from utils import build_module

# TODO: Here you should create your models. You can use the MLPModel or ConstantVelocity as a template.
#  Each model should have a __init__ function, a forward function, and a loss_function function.
#  The loss function doen't have to be in the model, but it is convenient to have it there, because the lit_module
#  will call it automatically, because you assign a prediction model to it and later it asks the model for the loss function.
import torch.nn as nn

# Not a NN 
class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        # x = x[:,-1,1:5]

        x_center = x[:,-1, 2]
        y_center = x[:,-1, 3]
        x_vel = x[:, -1, 5]
        y_vel = x[:, -1, 6]


        x_center_plus = x_center + self.dt * x_vel
        y_center_plus = y_center + self.dt * y_vel

        x_plus = torch.stack([x_center_plus,y_center_plus,x_vel, y_vel], dim=1)
        return x_plus

class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantAccelerationModel, self).__init__()
        self.dt = dt

    def forward(self, x):

        x_center = x[:, -1, 2]
        y_center = x[:, -1, 3]
        x_vel = x[:, -1, 5]
        y_vel = x[:, -1, 6]
        x_acc = x[:, -1, 7]
        y_acc = x[:, -1, 8]

        x_center_plus = x_center + self.dt * x_vel + (1/2)*(x_acc)^2
        y_center_plus = y_center + self.dt * y_vel + (1/2)*(y_acc)^2

        x_plus = torch.stack([x_center_plus, y_center_plus, x_vel, y_vel], dim=1)
        return x_plus

# this will be a neural network by itself
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        x = x.view(batch_size, -1, self.output_dim)
        return x


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size) -> None:
        """
        input_size: Number of features of your input vector
        hidden_size: Number of hidden neurons
        output_size: Number of features of your output vector
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns computed output and tanh(i2h + h2h)
        Inputs
        ------
        x: Input vector
        hidden_state: Previous hidden state
        Outputs
        -------
        out: Linear output (without activation because of how pytorch works)
        hidden_state: New hidden state matrix
        """
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)
        out = self.h2o(hidden_state)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
		Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)
    
class LSTMModel(nn.Module):
    def __init__(self, vocab_size=33278, ninp=200, nhid=200, nlayers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, vocab_size)

        self.nlayers = nlayers
        self.nhid = nhid
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # emb = self.drop(self.encoder(input))
        # output, hidden = self.rnn(emb, hidden)
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )