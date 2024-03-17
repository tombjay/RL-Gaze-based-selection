import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import gin
from helpers import*
from gazeenv import GazeEnv
import logging
from parameters import*

class Trainer(object):
    def __init__(self, model, timesteps, log_dir, log_dir_run, run, env, n_episodes):
        self.model = model
        self.timesteps = timesteps
        self.log_dir = log_dir
        self.log_dir_run = log_dir_run
        self.run = run
        self.env = env
        self.n_episodes = n_episodes
    
    
    def train(self):
        logging.info('Training Initiated..............')
        self.model.learn(total_timesteps = self.timesteps)
        model_path = os.path.join('saved-models', f'PPO_Gaze_Model{self.timesteps}trained')
        self.model.save(model_path)
        
        # plot learning curve
        plot_learning_curve(self.log_dir_run)
        plt.savefig(f'{self.log_dir}learning_curve{self.run}.png')
        plt.close('all')
    
    #test the model    
    def test(self, n_episodes):
        n_saccades = np.ndarray(shape=(n_episodes,1), dtype=np.float64)
        eps = 0
        while eps < n_episodes:                
            done=False
            step=0
            belief = self.env.reset(seed=None)
            while not done:
                step+=1
                action, _ = self.model.predict(belief, deterministic = True)
                belief, reward, done, truncated, info = self.env.step(action)
                if done:
                    n_saccades[eps]=step
                    eps+=1
                    break
            logging.info('Training completed..............{step}')

        np.savetxt( f'{self.log_dir_run}/num_saccades.csv', n_saccades, delimiter=',')