'''
This is an utils file to help some calculations during training and testing.
'''

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

# Reward smoothing during training.
def reward_smoothing(values, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_values = np.convolve(values, weights, 'valid')
    return smoothed_values

# To plot the learning curve.
def plot_learning_curve(log_folder, title='Learning Curve'):
    x_values, y_values = ts2xy(load_results(log_folder), 'timesteps')
    smoothed_rewards = reward_smoothing(y_values, window_size=100)
    trimmed_x_values = x_values[len(x_values) - len(smoothed_rewards):]
    
    figure = plt.figure(title)
    plt.plot(trimmed_x_values, smoothed_rewards)
    plt.xlabel('Timestep')
    plt.ylabel('Smoothed Rewards')

# Function to calculate distance between two points.
def dist_calculator(point_a, point_b):
    return np.linalg.norm(point_a - point_b)

def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)
        
def calc_saccade_time(amplitude):
    return 2.7 * amplitude + 37

def calc_ID(W, D):
    return np.log2(2*D/W)