import gin
import gymnasium as gym
import math
import os
import wandb
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gazeenv import GazeEnv
from train import Trainer
import argparse
from helpers import*
from parameters import*


parser = argparse.ArgumentParser()
parser.add_argument('--cross_validation', default='train', help='Specify whether to train or test a model. #train, #test')
parser.add_argument('--model_name', default='core_PPO', help='Specify the model to be trained. #core_PPO')
parser.add_argument('--train_mode', default='static', help='Specify the mode to be trained. #static, #random_init')
args = parser.parse_args()

def main(width, distance, ocular_noise, spatial_noise, timesteps, log_path, n_episodes, run):
    
    #wandb Initialization
    wandb.init(project="Gaze-based-selections", entity="jayakumarsundar-rs", sync_tensorboard=True)
    
    log_dir_train = f'{log_path}/train/w{width}d{distance}ocular{ocular_noise}spatial{spatial_noise}/'
    log_dir_run_train = f'{log_dir_train}/run{run}/'
    log_dir_test = f'{log_path}/test/w{width}d{distance}ocular{ocular_noise}spatial{spatial_noise}/'
    log_dir_run_test = f'{log_dir_test}/run{run}/'
    
    os.makedirs(log_dir_run_train, exist_ok=True)
    os.makedirs(log_dir_run_test, exist_ok=True)
    
    # To log the monitor env data.
    # Use accordingly for train & test
    monitor_log_train = log_dir_run_train
    monitor_log_test = log_dir_run_test
    
    # Env initialization
    env = GazeEnv(width, distance, ocular_noise, spatial_noise)
    
    # The monitor log path is different for train and test.
    # Kindly use the correct path.
    env = Monitor(env, monitor_log_test)
    
    # Model
    if args.model_name == "core_PPO":
        model = PPO('MlpPolicy', env, verbose=1, clip_range = 0.15, tensorboard_log=r'D:\Research-Project\infotech2023_jayakumar\tensorboard_logs')
    
    if args.cross_validation == "train":
        if args.model_name == "core_PPO":
            trainer = Trainer(model, timesteps, log_dir_train, log_dir_run_train, run, env, n_episodes)
            trainer.train()
    
    elif args.cross_validation == "test":
        model = PPO.load('D:\Research-Project\infotech2023_jayakumar\saved-models\PPO_Gaze_Model_4L_trained_all.zip', env = env)
        trainer = Trainer(model, timesteps, log_dir_test, log_dir_run_test, run, env, n_episodes)
        trainer.test(n_episodes)
    
    
if __name__ == "__main__":
    #loop to train and test all possible width & distance.
    if args.train_mode == "static":
        for w in width:
            for d in distance:        
                for model_runs in [10]: # To monitor the number of runs the training/testing has been done. 
                    main(w, d, ocular_noise, spatial_noise, timesteps, log_path, n_episodes, model_runs)
                
    # To train models with random initialization.
    elif args.train_mode == "random_init":
        combinations = [(w, d) for w in width for d in distance]
        for epoch in range(num_epochs):
            random.shuffle(combinations)  # Shuffle the combinations for each epoch
            num_batches = len(combinations) // batch_size

            for batch_idx in range(num_batches):
                batch = combinations[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                # Perform training on the selected batch
                for w, d in batch:
                    # Perform your training steps with the selected combination (w_val, d_val)
                    main(w, d, ocular_noise, spatial_noise, timesteps, log_path, n_episodes, runs = 10)
                    print(f"Training with width={w}, distance={d}")
    