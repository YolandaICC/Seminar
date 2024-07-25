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
        # Select the last time step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        # Extract features from the last time step
        x_center = x_last[:, 0]  # xCenter
        y_center = x_last[:, 1]  # yCenter
        x_velocity = x_last[:, 2]  # xVelocity
        y_velocity = x_last[:, 3]  # yVelocity
        x_acceleration = x_last[:, 4]  # xAcceleration
        y_acceleration = x_last[:, 5]  # yAcceleration

        # Calculate new positions based on constant acceleration model
        new_x_center = x_center + x_velocity * self.dt + 0.5 * x_acceleration * self.dt ** 2
        new_y_center = y_center + y_velocity * self.dt + 0.5 * y_acceleration * self.dt ** 2

        # Create new positions tensor with the shape (batch_size, 2)
        new_positions = torch.stack(
            (new_x_center, new_y_center, x_velocity, y_velocity, x_acceleration, y_acceleration),
            dim=1)  # shape: (batch_size, 6)

        return new_positions


class SingleTrackModel(nn.Module):
    def __init__(self, dt=1.0):
        super(SingleTrackModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        # Select the last time step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        # Extract features from the last time step
        x_center = x_last[:, 0]  # xCenter
        y_center = x_last[:, 1]  # yCenter
        heading = x_last[:, 2]  # heading
        x_velocity = x_last[:, 3]  # xVelocity
        y_velocity = x_last[:, 4]  # yVelocity

        # Calculate the change in position considering the heading
        delta_x = (x_velocity * torch.cos(heading) - y_velocity * torch.sin(heading)) * self.dt
        delta_y = (x_velocity * torch.sin(heading) + y_velocity * torch.cos(heading)) * self.dt

        # For single track model, the new position is old position plus the calculated delta
        new_x_center = x_center + delta_x
        new_y_center = y_center + delta_y

        # Create new positions tensor with the shape (batch_size, 2)
        new_positions = torch.stack((new_x_center, new_y_center, heading, x_velocity, y_velocity),
                                    dim=1)  # shape: (batch_size, 5)

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
        x = x.view(batch_size, -1, self.output_dim)
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
        output, (h_n, c_n) = self.lstm(x)
        # batch_size = x.shape[0]
        # x = x.flatten(start_dim=1)
        x, _ = self.lstm(x)
        x = self.linear(x)
        tensor_reduced = x[:, 0, :]
        return tensor_reduced


class HybridModel(nn.Module):
    def __init__(self, dt=1.0):
        super(HybridModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        # Select the last tim step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        x_center = x_last[:, 0]
        y_center = x_last[:, 1]
        x_vel = x_last[:, 2]
        y_vel = x_last[:, 3]

        # old position + velocity * dt
        new_x_center = x_center + self.dt * x_vel
        new_y_center = y_center + self.dt * y_vel
        new_positions = torch.stack((new_x_center, new_y_center, x_vel, y_vel),
                                    dim=1)  # shape: (batch_size, 2)

        return new_positions