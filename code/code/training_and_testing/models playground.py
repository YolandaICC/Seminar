import numpy as np
import pandas as pd
import pickle
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt

Track_Id = 281

tracks = pd.read_csv('/home/zil57ceb/Seafile/Mi Biblioteca/CAS Seminar Database/code/code/training_and_testing/dataset/data/00_tracks.csv')
tracks_Meta = pd.read_csv('/home/zil57ceb/Seafile/Mi Biblioteca/CAS Seminar Database/code/code/training_and_testing/dataset/data/00_tracksMeta.csv')
#print(tracks_Meta.head)

Id_0Meta_raw = tracks_Meta[(tracks_Meta["trackId"] == Track_Id)]
Id_0_raw = tracks[(tracks["trackId"] == Track_Id)]

Id_0 = Id_0_raw.reset_index()
Id_0Meta = Id_0Meta_raw.reset_index()
print("*************************************************")

# Get relevant info
x = Id_0["xCenter"]
y = Id_0["yCenter"]
xvel = Id_0["xVelocity"]
yvel = Id_0["yVelocity"]
vclass = Id_0Meta["class"]



# Constant velocity model
dt = 0.1
x_plus = []
y_plus = []
clase = Id_0Meta["class"][0]
print("***********************************")
for i in range(len(Id_0)):

    newx = x[i] + dt * xvel[i]
    newy = y[i] + dt * yvel[i]

    x_plus.append(newx)
    y_plus.append(newy)

# Plotting the trajectory
plt.figure(figsize=(10, 8))
plt.plot(x, y, label='Measured Vehicle Path')
plt.plot(x_plus, y_plus, label='Predicted Vehicle Path')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Vehicle Trajectory. Id: ' + str(Track_Id) + (" ") + str(clase))
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
