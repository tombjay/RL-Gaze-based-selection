'''
This file contains the code to render the environment frames into a video.
Run this as a standalone file with mode = 'rgb_array' in environment's render().
'''

import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display
import os
from gazeenv import GazeEnv
from JitterGazeEnv import JitterGazeEnv
import pyvirtualdisplay
import imageio
from stable_baselines3 import PPO

# Set up a virtual display

model_load_path = r'D:\Research-Project\infotech2023_jayakumar\saved-models\PPO_Gaze_Model5000000trained.zip'
if os.path.exists(model_load_path):
    model = PPO.load(model_load_path)

#set the env variable
env = GazeEnv(5, 10, 0.07, 0.09)

# Set up recording parameters
output_path = r'D:\Research-Project\infotech2023_jayakumar\figures\episodeimages\gaze-render-30fps.mp4'
frame_rate = 30

# Create a list to store frames
frames = []

# Loop to record frames
#for i in range(1000):
episodes = 400
for episode in range(1, episodes+1):
    belief = env.reset(seed=None)
    done = False
    score = 0

    #for t in range(100000):
    while not done:
        action, _ = model.predict(belief, deterministic = True)
        belief, reward, done, truncated, info = env.step(action)

        #if t % 10 == 0:
        frame = env.render()
        frames.append(frame)

# Save frames as an MP4 video using imageio
imageio.mimsave(output_path, frames, fps=frame_rate)