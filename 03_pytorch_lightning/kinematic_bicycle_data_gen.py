import pickle
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt


def bicycle_model(dt, num_steps, v, delta, L, x0, y0, theta0):
    """
    Simulates the bicycle model.

    :return: Arrays of x, y positions and headings theta.
    """
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    theta = np.zeros(num_steps)

    x[0], y[0], theta[0] = x0, y0, theta0

    for i in range(1, num_steps):
        x[i] = x[i - 1] + v * np.cos(theta[i - 1]) * dt
        y[i] = y[i - 1] + v * np.sin(theta[i - 1]) * dt
        theta[i] = theta[i - 1] + (v / L) * np.tan(delta[i]) * dt

    return x, y, theta


def save_as_pickle(data):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    signal_path = os.path.join(current_dir, 'data')
    if not os.path.exists(signal_path):
        os.makedirs(signal_path)

    with open(signal_path + '/kinematic_bicycle_model_signals.pkl', 'wb') as f:
        pickle.dump(data, f)


def find_data_min_max(results):

    # Convert the flattened list of tuples into a NumPy array
    data_array = np.array(results)

    # Determine the number of dimensions in the data
    num_dimensions = data_array.shape[1]

    # Initialize min and max lists
    min_values = []
    max_values = []

    # Loop through each dimension to find min and max
    for dim in range(num_dimensions):
        min_values.append(data_array[:, dim].min())
        max_values.append(data_array[:, dim].max())

    min_max_values = tuple(zip(min_values, max_values))

    # Return min and max values as a tuple of tuples
    return min_max_values


def save_min_max_to_json(min_max_values, file_path):
    # Prepare a dictionary to encode as JSON
    data = {'dimensions': [{'min': v[0], 'max': v[1]} for v in min_max_values]}
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    # Parameters
    dt = 0.1  # seconds
    total_time = 100  # seconds
    v = 5.0  # m/s, constant speed

    L = 2.0  # meters, wheelbase
    x0, y0, theta0 = 0, 0, 0  # initial conditions

    num_steps = int(total_time / dt)
    delta = np.zeros(num_steps)
    for i in range(num_steps):
        random_value = random.random()
        delta[i] = np.radians(random_value * 10)  # steering angle in radians

    # Generate data
    x, y, theta = bicycle_model(dt, num_steps, v, delta, L, x0, y0, theta0)

    data = np.concatenate([x[:, None], y[:, None], theta[:, None]], axis=1)
    save_as_pickle(data)

    # Find Min and Max values
    min_max_values = find_data_min_max(data)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    signal_path = os.path.join(current_dir, 'data')

    # Save Min and Max values to a JSON file
    min_max_values_path = os.path.join(signal_path, 'min_max_values.json')
    save_min_max_to_json(min_max_values, min_max_values_path)

    # Plotting the trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Bicycle Path')
    plt.scatter(x[0], y[0], color='red', label='Start')
    plt.scatter(x[-1], y[-1], color='green', label='End')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Bicycle Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
