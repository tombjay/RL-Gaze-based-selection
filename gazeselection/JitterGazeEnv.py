'''
This file holds the custom environment with jitter adaptations. 
Import the class and use it for training and testing the model.

Calculation of AR

Average radius = [4.5, 5.3, 6.0, 7.1, 7.9] in pixels 
Width = [40, 56, 70, 86, 100] in pixels

Convert it to Degrees of visal angle

Degree = Pixels/ Pixels per degree

Pixels per degree = Screen Width / (2* tan(FOV/2))

FOV = 2* arctan((Screen Diagonal)/ 2) / Viewing distance)

Hence the AR and Width can converted into 

AR = [0.14, 0.16, 0.19, 0.23, 0.25]

Width = [1.28, 1.79, 2.24, 2.74, 3.19]

[(0.14, 1.28), (0.16, 1.79), (0.19, 2.24), (0.23, 2.74), (0.25, 3.19)]

After Regression, the AR values for our target sizes in pairs:(AR, Target size)
[(0.13, 1), (0.15, 1.5), (0.18, 2), (0.24, 3), (0.27, 4), (0.29, 5)] in degrees.


'''

import gymnasium as gym
import math
import time
import numpy as np
import pygame.surfarray as surfarray
import pygame
from stable_baselines3.common.env_checker import check_env
from pygame.locals import *
from helpers import*
from parameters import*

class JitterGazeEnv(gym.Env):
    def __init__(self, width, distance, ocular_noise, spatial_noise):
        self.width_deg = width
        self.distance_deg = distance
        self.width = math.radians(width) # Diameter of target in radians
        self.distance = math.radians(distance) # Distance from start to target in radians
        self.ocular_noise = ocular_noise # Oculomotor noise
        self.spatial_noise = spatial_noise # Visual spatial noise
        self.theta = None # Angle of target from start position
        self.max_fixations = max_fixations # Maximum number of saccades allowed
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64) # Gaze fixation
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64) # Current gaze position
        self.state_space = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64) # Target location
        self.belief_space = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
        self.reward_range = (-1, 0) # Reward range
        self.AR = average_radius* self.width #From Paper. Allocated in parameters.py
        self.is_fixated = False
        self.fixation_time = 0
        self.jitter_position = None
        self.eye_movement_time = 0
        self.dwell_time_start = None
        self.selection_time = 0
        self.prev_pos = np.array([0, 0])
        self.onset_latency = target_onset_latency
        self.n_saccade = 1
        self.first_saccade_time = []
        self.second_saccade_time = []
        self.third_saccade_time = []
        self.rem_saccade_time = []
        self.emt_mean = np.zeros((4,1))
        self.is_dist_btw_jitter_gaze_and_target = False
        self.dist_btw_jitter_gaze_and_target = 0
        self.extra_saccades = 0
        self.jitter_shifts = 0
        self.iterator = 0
        self.extra_time = []
        self.gaze_positions = []
        
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
        self.is_fixated = False
        self.fixation_time = 0
        self.jitter_position = None
        self.info = {'target_pos': self.target_pos,
                     'current_pos': self.current_pos}
        self.eye_movement_time = 0
        self.dwell_time_start = None
        self.selection_time = 0
        self.prev_pos = np.array([0, 0]) 
        self.n_saccade = 1
        self.first_saccade_time = []
        self.second_saccade_time = []
        self.third_saccade_time = []
        self.rem_saccade_time = []
        self.is_dist_btw_jitter_gaze_and_target = False
        self.dist_btw_jitter_gaze_and_target = 0
        self.extra_saccades = 0
        self.jitter_shifts = 0
        self.iterator = 0
        self.extra_time = []
        self.gaze_positions = []
        
        return self.belief, self.info
        
    def force_gaze_shift(self):
        # Force gaze to shift using jitter_noise
        jitter_noise = np.random.normal(0, self.AR,  np.array(self.jitter_position).shape)
        self.current_pos = np.clip(self.jitter_position + jitter_noise, -1, 1)
        self.dist_btw_jitter_gaze_and_target = dist_calculator(self.current_pos, self.target_pos)
        check_dist = self.dist_btw_jitter_gaze_and_target < self.width / 2
        return (check_dist)
    
    def step(self, action):
        saccade_movement = dist_calculator(self.current_pos, action)
        noise_std = np.random.normal(0, self.ocular_noise * saccade_movement, np.array(action).shape)  # Generate random noise from Gaussian distribution
        self.current_pos = np.clip(action + noise_std, -1, 1)

        # Check the distance between gaze & target.
        dist_btw_gaze_and_target = dist_calculator(self.current_pos, self.target_pos)
        
        # Calculate distance covered in the saccade from the previous position to the current position.
        dist_covered = dist_calculator(self.prev_pos, self.current_pos)
        
        # Update prev_pos with the current position for the next step.
        self.prev_pos = self.current_pos

        # Calculate the eye movement time using the provided formula.
        self.eye_movement_time += calc_saccade_time(dist_covered)
        
        if self.num_fixations == 1:
            self.first_saccade_time.append(self.eye_movement_time)
        elif self.num_fixations == 2:
            self.second_saccade_time.append(self.eye_movement_time)
        else:
            self.third_saccade_time.append(self.eye_movement_time)   
        
        self.rem_saccade_time.append(self.eye_movement_time)

        # Check whether gaze is inside target region.
        if not self.is_fixated and dist_btw_gaze_and_target < self.width / 2:
            self.is_fixated = True
            self.jitter_position = self.current_pos
            
            # One time counter to calculate overall time to complete an episode.
            if self.iterator < 1:
                self.overall_time = time.time()
                self.iterator += 1

        if self.is_fixated:
            # Start second tie counter which is resettable.
            if self.dwell_time_start is None:
                self.dwell_time_start = time.time()
            
            # Forced gaze shifts.
            if self.jitter_shifts <= 7:
                jitter_noise = np.random.normal(0, self.AR,  np.array(self.jitter_position).shape)
                self.current_pos = np.clip(self.jitter_position + jitter_noise, -1, 1)
                self.jitter_shifts += 1
            
            self.dist_btw_jitter_gaze_and_target = dist_calculator(self.current_pos, self.target_pos)
            self.is_dist_btw_jitter_gaze_and_target = self.dist_btw_jitter_gaze_and_target < self.width / 2
            
            if self.is_dist_btw_jitter_gaze_and_target:
                jitter_time = time.time() - self.dwell_time_start
                
                # To check the end of an episode.
                if jitter_time >= 0.8:
                    done = True
                    reward = 0
                    truncated = False
                    self.selection_time = self.eye_movement_time + 800
                    self.extra_time = time.time() - self.overall_time
                    
                    '''
                    Considering a real-world scenario, it's unlikely that all eight forced jitters 
                    would consistently lead the gaze out of the target area. 
                    Therefore, if the gaze position deviates from the target more than twice, 
                    we apply a single deduction of 800ms (minimum dwell time). 
                    This adjustment ensures that the calculated selection times align more closely with actual human behavior.
                    '''
                    if self.extra_saccades == 0:
                        self.extra_time = 0
                    elif self.extra_saccades <= 2: 
                        self.extra_time = (self.extra_time*1000) - self.selection_time
                    else:
                        self.extra_time = (self.extra_time*1000) - self.selection_time - 800
                    
                else:
                    done = False
                    reward = 0
                    truncated = False
                    self.observation, self.observation_uncertainty = self.get_observation()
                    self.belief, self.belief_uncertainty = self.get_belief()
            
            else:
                self.dwell_time_start = None
                done = False
                truncated = False
                reward = -1
                self.is_fixated = False
                self.n_saccade += 1
                self.extra_saccades += 1
                self.observation, self.observation_uncertainty = self.get_observation()
                self.belief, self.belief_uncertainty = self.get_belief()
                
        else:
            done = False
            truncated = False
            reward = -1
            self.n_saccade += 1
            self.num_fixations += 1
            self.observation, self.observation_uncertainty = self.get_observation()
            self.belief, self.belief_uncertainty = self.get_belief()

        if self.num_fixations > self.max_fixations:
            done = True
            truncated = True
        
        # dict to store other essential values.
        addon_dict = {'num_fixation': self.num_fixations,
                      'eye_move_time': self.eye_movement_time,
                      'rem_saccade_time': self.rem_saccade_time,
                      'selection_time': self.selection_time,
                      'Num_saccade': self.n_saccade,
                      'Extra_saccades': self.extra_saccades,
                      'Extra_time': self.extra_time}

        return self.belief, reward, done, truncated, addon_dict
    
    # To get the partial observation.
    def get_observation(self):
        gaze_displacement = dist_calculator(self.target_pos,self.current_pos) #gaze eccentricity
        observation_uncertainty = gaze_displacement
        spatial_noise=np.random.normal(0, self.spatial_noise*gaze_displacement, self.target_pos.shape) # visual spatial noise is calculated by gaze & target eccentricity.
        observation=np.clip(self.target_pos + spatial_noise, -1, 1)
        
        return observation, observation_uncertainty

    # To get the belief
    def get_belief(self):
        new_observation, new_observation_uncertainity = self.observation, self.observation_uncertainity
        prev_belief, prev_belief_uncertainity = self.belief, self.belief_uncertainity
        
        scale_factor = pow(prev_belief_uncertainity, 2) / (pow(prev_belief_uncertainity, 2) + pow(new_observation_uncertainity, 2))
    
        new_belief = prev_belief + (scale_factor * (new_observation - prev_belief))
        new_belief_uncertainity = pow(prev_belief_uncertainity, 2) - (scale_factor * pow(prev_belief_uncertainity, 2))
        
        return new_belief, new_belief_uncertainity

    def render(self, mode='rgb_array'):
        # Screen dimensions
        screen_width = 1080
        screen_height = 720

        # World dimensions (to be consistent with (x, y) coordinates of the agent's position)
        world_width = 2
        world_height = 2

        # Scale factor for converting world coordinates to screen pixels
        scale = screen_width / world_width

        # Fill the screen with white
        self.screen.fill((255, 255, 255))

        # Target position in screen pixels
        target_pos_pix = np.array([
            int(self.target_pos[0] * scale + screen_width / 2),
            int(self.target_pos[1] * scale + screen_height / 2)
        ])

        # Current gaze position in screen pixels
        current_pos_pix = np.array([
            int(self.current_pos[0] * scale + screen_width / 2),
            int(self.current_pos[1] * scale + screen_height / 2)
        ])

        # Draw target position
        target_radius = int(self.width * scale / 2)
        pygame.draw.circle(self.screen, (255, 0, 0), target_pos_pix, target_radius)

        # Draw gaze position
        pygame.draw.circle(self.screen, (0, 0, 255), current_pos_pix, 10)

        # Draw lines to connect target and gaze positions
        pygame.draw.line(self.screen, (0, 0, 0), target_pos_pix, current_pos_pix, 2)
        
        #To nullify pygame not responding problem.
        pygame.event.get()

        # Display screen size, target size, and target distance
        font = pygame.font.Font(None, 24)
        screen_info_text = font.render(f'Screen Size: {screen_width} x {screen_height} pixels', True, (0, 0, 0))
        target_info_text = font.render(f'Target Size: {self.width_deg:.2f} degrees', True, (0, 0, 0))
        distance_info_text = font.render(f'Target Distance: {self.distance_deg:.2f} degrees', True, (0, 0, 0))
        fixation_info_text = font.render(f'Fixations: {self.num_fixations}', True, (0, 0, 0))
        AR_info_text = font.render(f'AR: {average_radius}', True, (0, 0, 0))
        selection_info_text = font.render(f'Selection Time (ms): {self.selection_time}', True, (0, 0, 0))
        self.screen.blit(screen_info_text, (10, 10))
        self.screen.blit(target_info_text, (10, 30))
        self.screen.blit(distance_info_text, (10, 50))
        self.screen.blit(fixation_info_text, (10, 70))
        self.screen.blit(selection_info_text, (10, 90))
        self.screen.blit(AR_info_text, (10, 110))

        # Update the display
        pygame.display.update()

        # Add a delay to visualize the gaze movement
        pygame.time.delay(200)
        
        if mode == 'human':
            return self.screen
        elif mode == 'rgb_array':
            image_array = surfarray.array3d(self.screen)
            return np.transpose(image_array, (1, 0, 2))  # Transpose dimensions to match matplotlib's default format
        else:
            raise ValueError(f"Unsupported rendering mode: {mode}")
    
    def render_and_save_image(self, episode, fixations, runs, image_size=(7*100, 5*100)):
        # Create an image surface for rendering
        image_surface = pygame.Surface((1080, 720))
        image_surface.fill((255, 255, 255))
        
        screen_width = 1080
        screen_height = 720
        world_width = 2
        scale = screen_width / world_width

        # Draw the target and gaze positions on the image surface
        target_pos_pix = np.array([
            int(self.target_pos[0] * scale + screen_width / 2),
            int(self.target_pos[1] * scale + screen_height / 2)
        ])
        current_pos_pix = np.array([
            int(self.current_pos[0] * scale + screen_width / 2),
            int(self.current_pos[1] * scale + screen_height / 2)
        ])
        target_radius = int(self.width * scale / 2)
        self.gaze_positions.append(np.copy(current_pos_pix))

        # Draw target position
        pygame.draw.circle(image_surface, (255, 0, 0), target_pos_pix, target_radius + 10)
        
        for i in range(1, len(self.gaze_positions)):
            pygame.draw.circle(image_surface, (0, 0, 0), self.gaze_positions[i], 3)
            pygame.draw.line(image_surface, (0, 0, 0), self.gaze_positions[i - 1], self.gaze_positions[i], 1)
        
        font = pygame.font.Font(None, 36)
        screen_info_text = font.render(f'Screen Size: {screen_width} x {screen_height} pixels', True, (0, 0, 0))
        target_info_text = font.render(f'Target Size: {self.width_deg:.2f} degrees', True, (0, 0, 0))
        distance_info_text = font.render(f'Target Distance: {self.distance_deg:.2f} degrees', True, (0, 0, 0))
        fixation_info_text = font.render(f'Fixations: {self.num_fixations}', True, (0, 0, 0))
        image_surface.blit(screen_info_text, (10, 10))
        image_surface.blit(target_info_text, (10, 40))
        image_surface.blit(distance_info_text, (10, 70))
        image_surface.blit(fixation_info_text, (10, 100))
        
        # Update the display
        pygame.display.update()

        # Save the image
        image_filename = f'D:\Research-Project\infotech2023_jayakumar\episodeimages\episode_{episode}_timestep_{fixations}_with_line.png'
        pygame.image.save(image_surface, image_filename)
        
        return self.gaze_positions