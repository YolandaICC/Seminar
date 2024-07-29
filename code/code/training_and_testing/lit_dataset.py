import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class inD_RecordingDataset(Dataset):
    def __init__(self, path, recording_id, sequence_length, item_type, features_tracks, features_tracksmeta,  train=True):
        """Dataset for inD dataset.
        Parameters
        ----------
        path : str
            Path to the data.
        recording_id : int
            Recording id of the data.
        sequence_length : int
            Length of the sequence.
        features : list
            List of features to use.
        train : bool
            Whether to use the training set or not.
        """
        super(inD_RecordingDataset).__init__()
        self.path = path
        self.recording_id = recording_id
        self.sequence_length = sequence_length
        self.features_tracks = features_tracks
        self.features_tracksmeta = features_tracksmeta
        self.item_type = item_type
        self.train = train
        self.transform = self.get_transform()
        if type(self.recording_id) == list:
            self.data = pd.DataFrame()
            tracks_data = pd.DataFrame()
            tracksMeta_data = pd.DataFrame()
            # TODO: Here we are simply loading the csv and stack them into one pandas dataframe.
            # You have to change this to load your data. This is just meant as a dummy example!!!
            for id in self.recording_id:
                with open(f"{path}/{id}_tracks.csv", 'rb') as f:
                    tracks_data = pd.concat([tracks_data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracks, dtype='float64')])

                with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
                    tracksMeta_data = pd.concat([tracksMeta_data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracksmeta)])

            # Label encoding - make to dtype = 'float64'
            # create a LabelEncoder object
            le = LabelEncoder()
            # Extract Precipitation Type as an array
            item_types = np.array(tracksMeta_data['class'])
            # label encode the 'Precip Type' column
            tracksMeta_data['class'] = le.fit_transform(item_types)
            # Left join with main table
            merged_data = tracks_data.merge(tracksMeta_data, on='trackId', how='left')

            # Data Normalization
            scaler = MinMaxScaler()
            columns_to_normalize = merged_data.columns[1:8]
            merged_data[columns_to_normalize] = scaler.fit_transform(merged_data[columns_to_normalize])
            merged_data = merged_data[(merged_data["class"] == self.item_type)]
            self.data = merged_data.head(100)


            encoded_values = list(le.classes_)
            actual_values = sorted(list(tracksMeta_data['class'].unique()))

            for i in range(len(encoded_values)):
                print(f'{actual_values[i]}: {encoded_values[i]}')


        else:
            self.data = pd.DataFrame()
            with open(f"{path}/{recording_id}_tracks.csv", 'rb') as f:
                tracks_data = pd.concat([self.data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracks, dtype='float64')])
                # self.data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features, dtype='str')

            with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
                tracksMeta_data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features_tracksmeta)
                # Label encoding - make to dtype = 'float64'
                # create a LabelEncoder object
                le = LabelEncoder()
                # Extract Precipitation Type as an array
                item_types = np.array(tracksMeta_data['class'])
                # label encode the 'Precip Type' column
                tracksMeta_data['class'] = le.fit_transform(item_types)
                # Left join with main table
                merged_data = tracks_data.merge(tracksMeta_data, on='trackId', how='left')

                # Data Normalization
                scaler = MinMaxScaler()
                columns_to_normalize = merged_data.columns[1:8]
                merged_data[columns_to_normalize] = scaler.fit_transform(merged_data[columns_to_normalize])
                merged_data = merged_data[(merged_data["class"] == self.item_type)]

                self.data = merged_data.head(100)

                # self.data = pd.concat([self.data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features, dtype='float64')])
                print(self.data)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
                 Returns the item at index idx.
        Parameters
        ----------
        idx : int
            Index of the item.
        Returns
        -------
        data : torch.Tensor
            The data at index idx.
        """
        if idx <= self.__len__():
            # for each step this function will be called
            data = self.data[idx:idx + self.sequence_length]
            # data type tensor specific for pytorh is like and array
            if self.transform:
                data = self.transform(np.array(data)).squeeze()
            return data
        else:
            print("wrong index")
            return None

    def get_transform(self):
        """
        Returns the transform for the data.
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return data_transforms