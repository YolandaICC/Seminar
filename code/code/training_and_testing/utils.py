import os
import datetime
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from pathlib import Path
import sys


def get_data_path():
    # TODO: change the path to your data path
    project_path = Path(__file__).resolve().parents[1]
    data_directory = 'training_and_testing/dataset/data'
    data_path = os.path.join(project_path, data_directory)
    sys.path.append(data_path)

    file_path = os.path.dirname(os.path.abspath(__file__))
    log_directory = 'logs'
    log_path = os.path.join(file_path, log_directory)
    return data_path, log_path


def create_wandb_logger(log_path, project_name, stage):
    date = get_current_datetime_str()
    wandb_logger = WandbLogger(project=project_name,
                               name=f"{project_name}_{stage}_{date}",
                               log_model=False,
                               version=date,
                               save_dir=log_path + "/wandb_logs")
    return wandb_logger


def get_current_datetime_str():
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d__%H_%M_%S")


def build_module(hidden_size, output_size, num_layers, pre_layers=None, droprate=0.3):
    layer_dims = torch.tensor(torch.linspace(hidden_size, output_size, num_layers), dtype=int)
    layers = nn.ModuleList()
    if pre_layers:
        if type(pre_layers) == list:
            for layer in pre_layers:
                layers.append(layer)
        else:
            layers.append(pre_layers)
    for dim in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[dim], layer_dims[dim + 1]))
        layers.append(nn.Dropout(droprate))
        layers.append(nn.Softplus(True))

    layers.append(nn.Linear(layer_dims[-1], output_size))
    return layers
