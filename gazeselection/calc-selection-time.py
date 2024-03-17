'''
This file contains the code to test the model by plotting the results indvidually.
'''

import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gazeenv import GazeEnv
from helpers import*
from parameters import*

class Calc_SelectionTime_NumSaccades(object):
    def __init__(self):
        self.ocular_noise = 0.07
        self.spatial_noise = 0.09
        self.width = np.array([1, 1.5, 2, 3, 4, 5])
        self.distance = np.array([5, 10])
        self.length_distance = len(self.distance)
        self.length_width = len(self.width)
        self.base_log_dir = r'D:\Research-Project\infotech2023_jayakumar\figures\selection-time'
        self.colors = ['#302594', '#942580', '#259439', '#944325']
        self.onset_latency = target_onset_latency         
    
    def calc_selection_time_width(self):
        for d in self.distance:
            fig=plt.figure(figsize=(7,5))
            time_mean = np.zeros((4, self.length_width))
            time_std = np.zeros((4, self.length_width))
            i  = 0
            
            for w in self.width:
                #Create lists to store individual saccade times.
                first_saccade_time = []
                second_saccade_time = []
                third_saccade_time = []
                rem_saccade_time = []
                
                #Instantiate the environment.
                env = GazeEnv(w, d, self.ocular_noise, self.spatial_noise)
                
                #Load the model to be tested.
                model_load_path = r'D:\Research-Project\infotech2023_jayakumar\saved-models\PPO_Gaze_Model_50L_trained.zip'
                if os.path.exists(model_load_path):
                    model = PPO.load(model_load_path)
                    
                #To get the saccade times.
                n_episodes = 50
                for episodes in range(0, n_episodes):
                    done = False
                    n_saccade = 0
                    belief, _ = env.reset(seed = None)
                    eye_movement_time = 0
                    current_gaze_pos = np.array([0,0])
                    
                    while not done:
                        n_saccade += 1
                        action, _ = model.predict(belief, deterministic = True)
                        belief, reward, done, truncated,  info = env.step(action)
                        dist_covered = dist_calculator(info['current_pos'], current_gaze_pos)
                        current_gaze_pos = info['current_pos']
                        eye_movement_time += calc_saccade_time(dist_covered) + self.onset_latency
                        if done:
                            if n_saccade == 1:
                                first_saccade_time.append(eye_movement_time)
                            elif n_saccade == 2:
                                second_saccade_time.append(eye_movement_time)
                            else:
                                third_saccade_time.append(eye_movement_time)
                            
                            rem_saccade_time.append(eye_movement_time)
                            break
                        
                time_mean[0,i] = np.mean(first_saccade_time)
                time_mean[1,i] = np.mean(second_saccade_time)
                time_mean[2,i] = np.mean(third_saccade_time)
                time_mean[3,i] = np.mean(rem_saccade_time)
                
                time_std[0,i] = np.std(first_saccade_time)
                time_std[1,i] = np.std(second_saccade_time)
                time_std[2,i] = np.std(third_saccade_time)
                time_std[3,i] = np.std(rem_saccade_time)
                
                i += 1
        
        w=np.array([1, 1.5, 2, 3, 4, 5])
        dwell_time=np.array([50,50,50,50,50,50])
        plt.errorbar(w,time_mean[0,:], yerr=time_std[0,:],fmt='d:', color=self.colors[0], elinewidth=1, capsize=2,label='1 saccade')
        plt.errorbar(w,time_mean[1,:], yerr=time_std[1,:],fmt='s:', color=self.colors[1], elinewidth=1, capsize=2,label='2 saccades')
        plt.errorbar(w,time_mean[2,:], yerr=time_std[2,:], fmt='<:', color=self.colors[2], elinewidth=1, capsize=2,label='3+ saccades')
        plt.errorbar(w,time_mean[3,:]+dwell_time, yerr=0, fmt='o-', color='k', elinewidth=1, capsize=2, label='selection time')
        plt.title('Model',fontsize=14)

        plt.xlabel(f'Target size ({chr(176)})',fontsize=14)
        plt.ylabel(f'Time (ms)',fontsize=14)
        plt.legend(title='Trials completed with:', loc='upper right',fontsize=14)
        plt.ylim([100 ,1200])
        plt.savefig(f'{self.base_log_dir}/selection-time-width-2-45.png')
        plt.show()
        plt.close()

    def calc_num_saccades(self):
        target_width = np.array([1, 1.5, 2, 3, 4, 5])
        plt.figure(figsize=(10,7))
        saccade_mean_d5 = []
        saccade_mean_d10 = []
        
        for d in self.distance:
            saccade_mean = []
            saccade_std = []
            
            for w in self.width:
                #Instantiate the environment.
                env = GazeEnv(w, d, self.ocular_noise, self.spatial_noise)
                
                #Load the model to be tested.
                model_load_path = r'D:\Research-Project\infotech2023_jayakumar\saved-models\PPO_Gaze_Model5000000trained.zip'
                if os.path.exists(model_load_path):
                    model = PPO.load(model_load_path)
                    
                #To get the saccade times.
                n_episodes = 50
                n_saccades = np.ndarray(shape=(n_episodes,1), dtype=np.float32)
                emt = np.ndarray(shape=(n_episodes,1), dtype=np.float32)
                eps = 0
                while eps < n_episodes: 
                    belief, _ = env.reset(seed = None)
                    done = False
                    n_saccade = 0
                    eye_movement_time = 0
                    current_gaze_pos = np.array([0,0])
                    
                    while not done:
                        n_saccade += 1
                        action, _ = model.predict(belief, deterministic = True)
                        belief, reward, done, truncated,  info = env.step(action)
                        dist_covered = dist_calculator(info['current_pos'], current_gaze_pos)
                        current_gaze_pos = info['current_pos']
                        eye_movement_time += 37 + 2.7 * dist_covered
                        
                        if done:
                            n_saccades[eps] = n_saccade
                            emt[eps] =  eye_movement_time
                            eps+=1
                            break
                        
                saccade_mean.append(np.round(np.mean(n_saccades),2))
                saccade_std.append(np.round(np.std(n_saccades),2))
                print(saccade_mean)

            saccade_mean_d5_50L = [2.06, 1.76, 1.38, 1.04, 1.0, 1.0] #one best run values
            saccade_mean_d10_50L = [2.66, 2.45, 1.95, 1.68, 1.3, 1.0]
            if d == 5:
                saccade_mean_d5 = saccade_mean 
                plt.plot(target_width, saccade_mean_d5_50L, 'd:',color='blue', label = f'Model D = {d}')
            elif d == 10:
                saccade_mean_d10 = saccade_mean
                plt.plot(target_width, saccade_mean_d10_50L, 's-',color='red', label = f'Model D = {d}')
                
            if d == 5:
                tmp = saccade_mean
            
            else:
                saccade_mean_poolled=(np.array(saccade_mean_d5_50L)+np.array(saccade_mean_d10_50L))/2
                print(saccade_mean_poolled)
                plt.plot(target_width, saccade_mean_poolled, 'ko-', label= f'Model (D=10 {chr(176)} and 5 {chr(176)} pooled)')
        
        
        plt.xlabel('Target size', fontsize=13)
        plt.ylabel('The number of saccade per trial', fontsize=13)
        plt.ylim(0.5, 6)
        plt.xticks(target_width)
        plt.legend(title='Num. of saccades', loc='upper right',fontsize=14)
        plt.savefig(r'D:\Research-Project\infotech2023_jayakumar\figures\selection-time/num_saccades-2-50.png')
        plt.show()
        return(saccade_mean_d5, saccade_mean_d10)
        
    def Index_of_Difficulty(self, saccade_mean_d5, saccade_mean_d10):
        plt.figure(figsize=(8, 5))
        W= np.array([1,1.5,2,3,4,5])

        ID1=calc_ID(W,5)
        ID2=calc_ID(W,10)
        '''
        If needed to plot I.D. directly from the saccade means, we could use the below two lines of code.
        Though, I have saved the best saccade means for different models.
        '''
        #plt.plot(ID1,saccade_mean_d5,'ro-',label='Distance=5')
        #plt.plot(ID2,saccade_mean_d10,'bs--',label='Distance=10')
        
        #Best Run Plot for 50Lakh timesteps
        saccade_mean_d5_50L = [2.06, 1.76, 1.38, 1.04, 1.0, 1.0] #one best run values
        saccade_mean_d10_50L = [2.66, 2.45, 1.95, 1.68, 1.3, 1.0] #one best run values
        plt.plot(ID1,saccade_mean_d5_50L,'ro-',label='Distance=5')
        plt.plot(ID2,saccade_mean_d10_50L,'bs--',label='Distance=10')
        
        #Best Run Plot for 60Lakh timesteps
        # saccade_mean_d5_60L = [1.78, 1.52, 1.24, 1.02, 1.0, 1.0] #one best run values
        # saccade_mean_d10_60L = [2.96, 3.0, 2.92, 2.68, 2.32, 1.56] #one best run values
        # plt.plot(ID1,saccade_mean_d5_60L,'ro-',label='Distance=5')
        # plt.plot(ID2,saccade_mean_d10_60L,'bs--',label='Distance=10')
        
        
        plt.xlabel('Index of difficulty (bits) ',fontsize=13)
        plt.ylabel('Saccades per trial',fontsize=13)
        plt.ylim([0.9, 3.5])
        plt.title('Model',fontsize=13)
        plt.legend(fontsize=10)
        plt.savefig(r'D:\Research-Project\infotech2023_jayakumar\figures\selection-time/ID-50lakhs.pdf')


if __name__ == "__main__":
    # Code block that will only execute when the script is run directly
    obj = Calc_SelectionTime_NumSaccades()
    obj.calc_selection_time_width()
    #obj.calc_num_saccades()
    #saccade_mean_d_5, saccade_mean_d_10 = obj.calc_num_saccades()
    #obj.Index_of_Difficulty(saccade_mean_d_5, saccade_mean_d_10)