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



