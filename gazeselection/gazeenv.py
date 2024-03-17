'''
This file holds the custom environment without jitter adaptations. 
Import the class and use it for training and testing the model.

'''
import gymnasium as gym
import math
import pygame
import numpy as np
import pygame.surfarray as surfarray
from stable_baselines3.common.env_checker import check_env
from pygame.locals import *
from helpers import*
from parameters import*


# Class to implement gaze environment.
class GazeEnv(gym.Env):
    def __init__(self, width, distance, ocular_noise, spatial_noise):
        self.width_deg = width
        self.distance_deg = distance
        self.width = math.radians(width) # Diameter of target in radians
        self.distance = math.radians(distance) # Distance from start to target in radians
        self.ocular_noise = ocular_noise # Oculomotor noise
        self.spatial_noise = spatial_noise # Visual spatial noise
        self.theta = None # Angle of target from start position
        self.current_pos = np.array([0, 0]) # Current gaze position or fixation
        self.max_fixations = max_fixations # Maximum number of saccades allowed. Allocated in parameters.py
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64) # Gaze fixation
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64) # Current gaze position
        self.state_space = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64) # Target location
        self.belief_space = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
        self.reward_range = (-1, 0) # Reward range
        
        #To visualise the env.
        pygame.init()
        self.screen = pygame.display.set_mode((1080, 720))
        self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        self.theta = np.random.uniform(low=0, high=2*np.pi)
        self.target_pos = np.array([self.distance * np.cos(self.theta), self.distance * np.sin(self.theta)])
        self.current_pos = np.array([0, 0])
        self.num_fixations = 1 # 1 because gaze is fixated at [0,0] initially
        self.observation, self.observation_uncertainity = self.get_observation()
        self.belief, self.belief_uncertainity = self.observation, self.observation_uncertainity #As per paper, belief is based on agent's observation
        self.info = {'target_pos': self.target_pos,
                     'current_pos': self.current_pos}
        
        return self.belief
    
    
    def step(self, action):
        saccade_movement = dist_calculator(self.current_pos, action)
        noise_std = np.random.normal(0, self.ocular_noise*saccade_movement, action.shape) #Generate random noise from gaussian distribution- ".normal(loc, scale, size)"" 
        self.current_pos = np.clip(action + noise_std, -1, 1)
        
        self.num_fixations += 1
        
        #Check the distance between gaze & target.
        dist_btw_gaze_and_target = dist_calculator(self.current_pos, self.target_pos)
        
        if dist_btw_gaze_and_target < self.width/2: #As per paper if distance is less half of target width, the gaze is in target region.
            reward = 0
            done = True
            truncated = False
        else:
            reward = -1
            done = False
            truncated = False
            self.observation, self.observation_uncertainity = self.get_observation()
            self.belief, self.belief_uncertainity = self.get_belief()
            
        if self.num_fixations > self.max_fixations:
            done = True
            truncated = True
            
        
        #dict to store other essential values.
        addon_dict = { 'target_pos' : self.target_pos,
                       'current_pos': self.current_pos,
                       'belief': self.belief,
                       'action': action,
                       'num_fixation': self.num_fixations}

        return self.belief, reward, done, truncated, addon_dict
    
    def get_observation(self):
        gaze_displacement = dist_calculator(self.target_pos,self.current_pos) #gaze eccentricity
        observation_uncertainty = gaze_displacement
        spatial_noise=np.random.normal(0, self.spatial_noise*gaze_displacement, self.target_pos.shape) # visual spatial noise is calculated by gaze & target eccentricity.
        observation=np.clip(self.target_pos + spatial_noise, -1, 1)
        
        return observation, observation_uncertainty
    
    def get_belief(self):
        new_observation, new_observation_uncertainity = self.observation, self.observation_uncertainity
        prev_belief, prev_belief_uncertainity = self.belief, self.belief_uncertainity
        
        
        scale_factor = pow(prev_belief_uncertainity, 2) / (pow(prev_belief_uncertainity, 2) + pow(new_observation_uncertainity, 2))
    
        new_belief = prev_belief + (scale_factor * (new_observation - prev_belief))
        new_belief_uncertainity = pow(prev_belief_uncertainity, 2) - (scale_factor * pow(prev_belief_uncertainity, 2))
        
        return new_belief, new_belief_uncertainity

    def render(self, mode='rgb_array'):
        
        # Fill the screen with white
        self.screen.fill((255, 255, 255))
        screen_width = 1080
        screen_height = 720
        
        world_width = 2 #To be consistent with (x,y) coordinates of agent's position
        world_height = 2
        scale = screen_width/world_width
        radius = int(self.width*scale/2)
        target_pos_pix = np.array([int(self.target_pos[0]*scale+screen_width/2), int(self.target_pos[1]*scale+screen_height/2)])
        current_pos_pix = np.array([self.current_pos[0]*scale+screen_width/2, self.current_pos[1]*scale+screen_height/2], dtype= int)
        
        #To nullify pygame not responding problem.
        pygame.event.get()
        
        #Draw target position
        pygame.draw.circle(self.screen, (255, 0, 0), target_pos_pix, radius)
        
        #Draw gaze position
        pygame.draw.circle(self.screen, (0, 0, 255), current_pos_pix, 10)
        pygame.draw.line(self.screen, (0, 0, 0), target_pos_pix, current_pos_pix, 2)
        
        font = pygame.font.Font(None, 24)
        screen_info_text = font.render(f'Screen Size: {screen_width} x {screen_height} pixels', True, (0, 0, 0))
        target_info_text = font.render(f'Target Size: {self.width_deg:.2f} degrees', True, (0, 0, 0))
        distance_info_text = font.render(f'Target Distance: {self.distance_deg:.2f} degrees', True, (0, 0, 0))
        fixation_info_text = font.render(f'Fixations: {self.num_fixations}', True, (0, 0, 0))
        self.screen.blit(screen_info_text, (10, 10))
        self.screen.blit(target_info_text, (10, 30))
        self.screen.blit(distance_info_text, (10, 50))
        self.screen.blit(fixation_info_text, (10, 70))
        
        # Update the display
        pygame.display.update()
        
        # Add a delay to visualize the gaze movement
        pygame.time.delay(200)
        
        if mode == 'human':
            return None
        elif mode == 'rgb_array':
            image_array = surfarray.array3d(self.screen)
            return np.transpose(image_array, (1, 0, 2))  # Transpose dimensions to match matplotlib's default format
        else:
            raise ValueError(f"Unsupported rendering mode: {mode}")