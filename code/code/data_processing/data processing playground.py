import numpy as np
import pandas as pd
import pickle
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

path_tracks = '/home/zil57ceb/Seafile/Mi Biblioteca/CAS Seminar Database/code/code/training_and_testing/dataset/data/00_tracks.csv'
path_tracks_Meta = '/home/zil57ceb/Seafile/Mi Biblioteca/CAS Seminar Database/code/code/training_and_testing/dataset/data/00_tracksMeta.csv'

data = pd.DataFrame()
data2 = pd.DataFrame()



data = pd.concat([data, pd.read_csv(path_tracks, delimiter=',', header=0, usecols= ["recordingId","trackId","xCenter","yCenter","xVelocity", "yVelocity",], dtype='float64')])
data2 = pd.concat([data2, pd.read_csv(path_tracks_Meta, delimiter=',', header=0, usecols= ["trackId","class"])])







# create a LabelEncoder object
le = LabelEncoder()

# Extract Precipitation Type as an array
item_types = np.array(data2['class'])

# label encode the 'Precip Type' column
data2['class'] = le.fit_transform(item_types)

# get the actual categorical values and their corresponding encoded values
encoded_values = list(le.classes_)
actual_values = sorted(list(data2['class'].unique()))

# print the actual values and their encoded values
for i in range(len(encoded_values)):
    print(f'{actual_values[i]}: {encoded_values[i]}')

data = data.merge(data2, on='trackId', how='left')

car_data = data[(data["class"] == 1)]

print (car_data)

if self.item_type == 0:
    self.data = merged_data[(merged_data["class"] == self.item_type)]
elif self.item_type == 1:
    self.data = merged_data[(merged_data["class"] == self.item_type)]
elif self.item_type == 2:
    self.data = merged_data[(merged_data["class"] == self.item_type)]
elif self.item_type == 3:
    self.data = merged_data[(merged_data["class"] == self.item_type)]




# /////////////////////////////////////////////////////////////////////////////
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from sklearn.preprocessing import LabelEncoder
#
# class inD_RecordingDataset(Dataset):
#     def __init__(self,
#                  path,
#                  recording_id,
#                  sequence_length,
#                  features_tracks,
#                  features_tracksMeta,
#                  item_type,
#                  train=True):
#         """Dataset for inD dataset.
#         Parameters
#         ----------
#         path : str
#             Path to the data.
#         recording_id : int
#             Recording id of the data.
#         sequence_length : int
#             Length of the sequence.
#         features : list
#             List of features to use.
#         train : bool
#             Whether to use the training set or not.
#         """
#         super(inD_RecordingDataset).__init__()
#         self.path = path
#         self.recording_id = recording_id
#         self.sequence_length = sequence_length
#         self.features_tracks = features_tracks
#         self.features_tracksMeta = features_tracksMeta
#         self.train = train
#         self.transform = self.get_transform()
#         self.item_type = item_type
#
#
#         if type(self.recording_id) == list:
#             self.data = pd.DataFrame()
#             # TODO: Here we are simply loading the csv and stack them into one pandas dataframe.
#             # You have to change this to load your data. This is just meant as a dummy example!!!
#             for id in self.recording_id:
#                 with open(f"{path}/{id}_tracks.csv", 'rb') as f:
#                     self.data = pd.concat([self.data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracks, dtype='float64')])
#
#                 with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
#
#                     tracksMeta_data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracksMeta)
#                     # Label encoding - make to dtype = 'float64'
#                     # create a LabelEncoder object
#                     le = LabelEncoder()
#                     # Extract Precipitation Type as an array
#                     item_types = np.array(tracksMeta_data['class'])
#                     # label encode the 'Precip Type' column
#                     tracksMeta_data['class'] = le.fit_transform(item_types)
#                     # Left join with main table
#                     self.data = self.data.merge(tracksMeta_data, on='trackId', how='left')
#                     # self.data = merged_data[(merged_data["class"] == self.item_type)]
#
#
#                     encoded_values = list(le.classes_)
#                     actual_values = sorted(list(tracksMeta_data['class'].unique()))
#                     for i in range(len(encoded_values)):
#                         print(f'{actual_values[i]}: {encoded_values[i]}')
#         else:
#             with open(f"{path}/{recording_id}_tracks.csv", 'rb') as f:
#                 self.data = pd.concat([self.data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracks, dtype='float64')])
#
#             with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
#                 tracksMeta_data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracksMeta)
#                 # Label encoding - make to dtype = 'float64'
#                 # create a LabelEncoder object
#                 le = LabelEncoder()
#                 # Extract Precipitation Type as an array
#                 item_types = np.array(tracksMeta_data['class'])
#                 # label encode the 'Precip Type' column
#                 tracksMeta_data['class'] = le.fit_transform(item_types)
#                 # Left join with main table
#                 self.data = self.data.merge(tracksMeta_data, on='trackId', how='left')
#                 # self.data = merged_data[(merged_data["class"] == self.item_type)]
#
#
#
#     def __len__(self):
#         """
#         Returns the length of the dataset.
#         """
#         return len(self.data) - self.sequence_length
#
#     def __getitem__(self, idx):
#         """
#                  Returns the item at index idx.
#         Parameters
#         ----------
#         idx : int
#             Index of the item.
#         Returns
#         -------
#         data : torch.Tensor
#             The data at index idx.
#         """
#         if idx <= self.__len__():
#             # for each step this function will be called
#             data = self.data[idx:idx + self.sequence_length]
#             # data type tensor specific for pytorh is like and array
#             if self.transform:
#                 data = self.transform(np.array(data)).squeeze()
#             return data
#         else:
#             print("wrong index")
#             return None
#
#     def get_transform(self):
#         """
#         Returns the transform for the data.
#         """
#         data_transforms = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#         return data_transforms