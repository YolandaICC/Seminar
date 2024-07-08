import numpy as np
import pandas as pd 

from readDataset import dataGrabber

dataset_path = 'C:\\Users\\yolis\\Documents\\CAS Seminar Database\\code\\code\\training_and_testing\\dataset\\data\\'

recording_id_sel = ['0','1','11']

# Initialize data Grabber Object
data_obj = dataGrabber(dataset_path)

data_obj.recording_id = recording_id_sel
data_obj.read_csv_with_recordingID()

track_data_raw = data_obj.get_tracks_data()
track_meta_data_raw = data_obj.get_tracksMeta_data()

track_data_raw