import pickle
import torch
import json
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms


class KinematicBicycleDataset(Dataset):
    def __init__(self, sequence_length):
        # Store the input data and transformations
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.signal_filename = 'kinematic_bicycle_model_signals.pkl'
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.pickle_path = os.path.join(self.current_dir, 'data', self.signal_filename)
        self.min_max_values_path = os.path.join(self.current_dir, 'data', 'min_max_values.json')
        self.sequence_length = sequence_length

        self.data_original = []
        self.data_normalized = []

        self.USE_NORMALIZED_DATA = True

        self.data_original, self.data_normalized = self.read_dataset_pickle(self.pickle_path, self.min_max_values_path)

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.data_original.shape[0] - self.sequence_length

    def __getitem__(self, index):
        if index <= self.__len__():
            if self.USE_NORMALIZED_DATA:
                item = self.data_normalized[index:index + self.sequence_length]
            else:
                item = self.data_original[index:index + self.sequence_length]

            return item
        else:
            raise IndexError("Index out of range")

    def read_min_max_from_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extracting min and max values into a list of tuples
        min_max_list = [(dim['min'], dim['max']) for dim in data['dimensions']]

        # Converting list of tuples to a NumPy array
        min_max_array = np.array(min_max_list)
        return min_max_array

    def normalize_data(self, x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    def read_dataset_pickle(self, pickle_path, min_max_values_path=None):
        data_normalized = []

        with open(pickle_path, 'rb') as f:
            data_original = pickle.load(f)

        if min_max_values_path:
            min_max_values = self.read_min_max_from_json(min_max_values_path)

            # Normalize the data
            x_data = self.normalize_data(data_original[:, 0], min_max_values[0][0], min_max_values[0][1])
            y_data = self.normalize_data(data_original[:, 1], min_max_values[1][0], min_max_values[1][1])
            theta = self.normalize_data(data_original[:, 2], min_max_values[2][0], min_max_values[2][1])

            data_normalized = self.transforms(np.column_stack((x_data, y_data, theta))).squeeze()
        data_original = self.transforms(data_original).squeeze()
        return data_original, data_normalized
