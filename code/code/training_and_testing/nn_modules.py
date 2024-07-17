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
        # Select the last tim step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        x_center = x_last[:, 0]
        y_center = x_last[:, 1]
        x_vel = x_last[:, 2]
        y_vel = x_last[:, 3]
        # x_acc = x[:, 7]
        # y_acc = x[:, 8]

        # old position + velocity * dt
        new_x_center = x_center + self.dt * x_vel
        new_y_center = y_center + self.dt * y_vel
        new_positions = torch.stack((new_x_center, new_y_center, x_vel, y_vel),
                                    dim=1)  # shape: (batch_size, 2)

        # Replicate the new_positions to match the number of features
        # new_positions = new_positions.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # shape: (batch_size, 2, number_of_features)

        return new_positions

class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantAccelerationModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        # Select the last tim step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        x_center = x[:, 0]
        y_center = x[:, 1]
        x_vel = x[:, 2]
        y_vel = x[:, 3]
        # x_acc = x[:, 7]
        # y_acc = x[:, 8]

        # old position + velocity * dt
        new_x_center = x_center + self.dt * x_vel
        new_y_center = y_center + self.dt * y_vel
        new_positions = torch.stack((new_x_center, new_y_center, x_vel, y_vel),
                                    dim=1)  # shape: (batch_size, 2)

        # Replicate the new_positions to match the number of features
        # new_positions = new_positions.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # shape: (batch_size, 2, number_of_features)

        return new_positions

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

        return x


class LSTMModel(nn.Module):
    r""" A simple LSTM model.
    This new version is seeking to eliminate the text prediction structure
    and be simple to handle Tensors' shape issues """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, future_sequence_length=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1)
        self.linear = nn.Linear(hidden_dim, output_dim)  # * future_sequence_length

    def forward(self, x):
        # output, (h_n, c_n) = self.lstm(x)
        # batch_size = x.shape[0]
        # x = x.flatten(start_dim=1)
        x, _ = self.lstm(x)
        x = self.linear(x)
        # x = x.view(batch_size, -1, self.output_dim)
        tensor_reduced = x[:, 0, :]
        return tensor_reduced

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters())
    #     hidden_tensor = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
    #                      weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
    #     return hidden_tensor


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

