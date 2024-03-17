'''
This file is substitue file for inbuilt rendering function of environment.
Run this as a standalone file with mode = 'rgb_array' in environment's render(). 
'''

import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display

from gazeenv import GazeEnv

VISUALIZE = True
RANDOM_SEED = 42

# Setup environment
env = GazeEnv(5, 10, 0.07, 0.09)
observation, info = env.reset()

if VISUALIZE:
    fig, ax = plt.subplots()  # Create a figure and axes for visualization

for i in range(1000):
    render_frame = env.render()

    if i == 0:
        print(f'State shape: {render_frame[0].shape}')  # Print the shape of the first returned value

    if VISUALIZE:
        ax.cla()  # Clear the previous plot
        ax.imshow(render_frame[0])  # Display the image (assuming the first returned value is an image)
        display.display(plt.gcf())
        display.clear_output(wait=True)

    action = env.action_space.sample()
    observation, reward, done, truncated,  info = env.step(action)

    if done:
        observation, info = env.reset()

env.close()





