'''

This is a standalone file to plot all the graphs that are used in the report.
Call the required function and the graph would be plotted accordingly under the Figures folder.

'''

import gin
import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from helpers import*

class plot_extraction():
    def __init__(self):
        self.colors = ['#302594', '#942580', '#259439', '#944325']
        self.fig_plot_path = r'D:\Research-Project\infotech2023_jayakumar\figures\selection-time' # Base directory to store the figures.
        self.width = np.array([1, 1.5, 2, 3, 4, 5])

    # To plot selection time from schuetz without jitters 
    def sel_time(self):
        fig=plt.figure(figsize=(10,5))
        # To plot baseline model.
        time_first = [220, 220, 220, 220, 220, 220]
        time_second = [410, 410, 410, 410, 410, 410]
        time_third = [780, 685, 670]
        time_fourth = [590, 425, 375, 300, 270, 260]
        
        
        plt.subplot(1,3,1)
        plt.errorbar(self.width,time_first, yerr = 0,fmt = 'd:', color = self.colors[0], elinewidth = 1, capsize = 2, label = '1 saccade')
        plt.errorbar(self.width,time_second, yerr = 0,fmt = 's:', color = self.colors[1], elinewidth = 1, capsize = 2, label = '2 saccades')
        plt.errorbar(self.width[0:3],time_third, yerr = 0, fmt = '<:', color = self.colors[2], elinewidth = 1, capsize = 2, label = '3 saccades')
        plt.errorbar(self.width,time_fourth, yerr = 0, fmt = 'o-', color = 'black', elinewidth = 1, capsize = 2, label = 'selection time')
        

        plt.title('Baseline Model',fontsize = 13)
        plt.xlabel(f'Target size ({chr(176)})',fontsize = 12)
        plt.ylabel('Time (ms)',fontsize=12)
        plt.xticks(self.width)
        #plt.legend(title='Trials completed with:', loc = 'upper right',fontsize=10)
        plt.ylim([100 ,1200])
        
        # To plot from Data.
        first_saccade_time_err = np.array([13, 13, 13, 13, 13, 13])
        first_saccade_time = np.array([228, 224, 224, 228, 228, 228])
        second_saccade_time_err = np.array([22, 30, 34, 73, 56, 64])
        second_saccade_time = np.array([400, 413, 422, 448, 439, 353])
        third_saccade_time_err = np.array([159, 133, 107])
        third_saccade_time = np.array([909, 909, 750])
        selection_time_err = np.array([30, 22, 17, 21, 17, 9])
        selection_time = np.array([560, 439, 336, 301, 241, 258])


        plt.subplot(1,3,2)
        plt.errorbar(self.width, first_saccade_time, yerr = first_saccade_time_err,fmt = 'd:', color = self.colors[0], elinewidth = 1, capsize = 2,label = '1 saccades')
        plt.errorbar(self.width, second_saccade_time, yerr = second_saccade_time_err,fmt = 's:', color = self.colors[1], elinewidth = 1, capsize = 2,label = '2 saccades')
        plt.errorbar(self.width[0:3], third_saccade_time, yerr = third_saccade_time_err,fmt = '<:', color = self.colors[2], elinewidth = 1, capsize = 2,label = '3+ saccades')
        plt.errorbar(self.width, selection_time, yerr = selection_time_err, fmt = 'o-', color = 'black', elinewidth = 1, capsize = 2, label = 'selection time')

        plt.title('Data',fontsize=13)
        #plt.legend(title='Trials completed with:', loc='upper right',fontsize=10)
        plt.xlabel(f'Target size ({chr(176)})',fontsize=12)
        plt.xticks(self.width)
        plt.ylim([100 ,1200])
        
        # To plot from our model.
        plt.subplot(1,3,3)
        time_first_new = [247.93558925, 248.07983924, 247.90749397, 247.88582058, 247.83712342, 247.80397471]
        time_second_new = [486.34321047, 485.97110728, 486.49863032, 487.24996096, 486.16075387, 487.37122775]
        time_third_new = [722.99879556, 724.40584642, 724]
        time_fourth_new = [519.26474804, 443.20502614, 367.20306214, 310.12049708, 257.37006864, 257.38666483]
        
        
        dwell_time=np.array([50,50,50,50,50,50])
        plt.errorbar(self.width,time_first_new, yerr = 0, fmt = 'd:', color = self.colors[0], elinewidth = 1, capsize = 2,label = '1 saccade')
        plt.errorbar(self.width,time_second_new, yerr = 0, fmt = 's:', color = self.colors[1], elinewidth = 1, capsize = 2,label = '2 saccades')
        plt.errorbar(self.width[0:3],time_third_new, yerr = 0, fmt = '<:', color = self.colors[2], elinewidth = 1, capsize = 2,label = '3 saccades')
        plt.errorbar(self.width,time_fourth_new + dwell_time, yerr = 0, fmt = 'o-', color = 'black', elinewidth = 1, capsize = 2,label = 'selection time')
        

        plt.title('Custom Model',fontsize=13)
        plt.xlabel(f'Target size ({chr(176)})',fontsize = 12)
        plt.xticks(self.width) 
        plt.legend(title='Trials completed with:', loc='upper right',fontsize = 10)
        plt.ylim([100 ,1200])
        plt.savefig(os.path.join(self.fig_plot_path, 'selection-time-width-combined.png'),bbox_inches='tight', dpi=300)
        plt.close()

    # To plot the selection time from jitters.
    def sel_time_jitter(self):
        '''
        Extra time is calculated manually, by inspecting the 'info' dictionary used
        in the 'step function' of 'jitter-env-trial.ipynb' and/or 'Jitter-GazeEnv.py' files.
        
        Dwell time = 800ms + avg(Extra time) for each target size.
        Final selction time = EMT + Dwell time
        '''
        fig = plt.figure(figsize=(8,6))
        ax = fig.gca()
        eye_movement_time = np.array([519.26474804, 443.20502614, 367.20306214, 310.12049708, 257.37006864, 257.38666483])
        dwell_time = np.array([1252, 1204, 1178, 1091, 1027, 1024]) # Extra time = [452, 404, 378, 291, 227, 224] in ms
        final_sel_time = np.array([1771, 1647, 1545, 1401, 1284, 1281])
        
        ax.bar(self.width, eye_movement_time, 0.3, color = 'red',label = 'Eye movement time')
        plt.bar(self.width, final_sel_time - eye_movement_time, 0.3, 
                bottom = eye_movement_time, color = '#348ceb', label = 'Selection time')
        
        for i in range(6):
            ax.text(self.width[i]-0.1,800,f'{dwell_time[i]}',rotation = 90,fontsize = 18)

        plt.xlabel(f'Target size({chr(176)})', fontsize = 15)
        plt.ylabel('Time(ms)', fontsize = 15)
        plt.xticks(self.width, fontsize = 12)
        plt.ylim([200,2000])
        plt.yticks(np.arange(0,2000,250), fontsize = 12)
        plt.legend(fontsize = 14)
        plt.title('Effect of Jitters')
        plt.savefig(os.path.join(self.fig_plot_path, 'selection-time-jitter.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

    # To plot the number of saccades. 
    def num_saccade(self):
        fig = plt.figure(figsize=(10,5))
        
        # To plot from baseline model.
        plt.subplot(1,3,1)
        saccade_mean_d5 = np.array([1.65, 1.38, 1.15, 1.0, 1.0, 1.0])
        saccade_mean_d10 = np.array([1.98, 1.78, 1.57, 1.35, 1.18, 1.0])
        plt.plot(self.width, saccade_mean_d5, linestyle = 'dotted', marker = 'o',color='blue', label = f'D = 5{chr(176)}')
        plt.plot(self.width, saccade_mean_d10, linestyle = 'dotted', marker ='s',color='red', label = f'D = 10{chr(176)}')

        saccade_mean_poolled = (np.array(saccade_mean_d5)+np.array(saccade_mean_d10))/2
        plt.plot(self.width, saccade_mean_poolled, linestyle = 'solid', marker = 'd', color = 'black', label = f'(D=10{chr(176)} and 5{chr(176)} pooled)')
        plt.title('Baseline Model', fontsize = 13)  
        plt.xlabel('Target size', fontsize = 13)
        plt.ylabel('The number of saccade per trial', fontsize = 13)
        plt.ylim(0.5, 4)
        plt.xticks(self.width)
        plt.legend(title = 'Num. of saccades', loc='upper right', fontsize = 10)
        
        # To plot from data.
        plt.subplot(1,3,2)
        num_fix = np.array([1.52034339, 1.48997083, 1.36647516, 1.13782123, 1.08996991, 1.05730888])
        err1 = np.array([0.16303721, 0.09226533, 0.09684957, 0.05644785, 0.03552923, 0.02865258])
        plt.errorbar(self.width, num_fix, yerr = err1, linestyle = 'solid', marker = 'd',
                    color = 'black',  capsize = 2,label = f'(D=10{chr(176)} and 5{chr(176)} pooled)')
        
        plt.title('Data',fontsize = 12)
        plt.xlabel('Target size', fontsize = 12)
        plt.ylim(0.5, 4)
        plt.xticks(self.width)
        plt.legend(title='Num. of saccades',fontsize = 10)
        
        # To plot from our model.
        plt.subplot(1,3,3)
        saccade_mean_d5 = np.array([2.06, 1.76, 1.38, 1.04, 1.0, 1.0])
        saccade_mean_d10 = np.array([2.66, 2.45, 1.95, 1.68, 1.3, 1.0])
        plt.plot(self.width, saccade_mean_d5, linestyle='dotted', marker='o', color='blue', label = f'D = 5{chr(176)}')
        plt.plot(self.width, saccade_mean_d10, linestyle='dotted', marker='s' ,color='red', label = f'D = 10{chr(176)}')

        saccade_mean_poolled=(np.array(saccade_mean_d5)+np.array(saccade_mean_d10))/2
        plt.plot(self.width, saccade_mean_poolled, linestyle='solid', marker='d', color='black', label= f'(D=10{chr(176)} and 5{chr(176)} pooled)')
        plt.title('Custom Model',fontsize=13)  
        plt.xlabel('Target size', fontsize=13)
        plt.ylim(0.5, 4)
        plt.xticks(self.width)
        plt.legend(title='Num. of saccades', loc='upper right',fontsize=10)
        plt.savefig(os.path.join(self.fig_plot_path, 'num_saccades-basemodel.pdf'), bbox_inches='tight', dpi=300)   

    # To plot ID against saccades.
    def ID(self):
        # To plot from baseline model.
        plt.figure(figsize=(10, 5))
        plt.subplot(1,3,1)
        num_saccade_D5 = np.array([1.63, 1.31, 1.12, 1.02, 1.0, 1.0])
        num_saccade_D10 = np.array([2.04, 1.82, 1.56, 1.22, 1.08, 1.02])

        ID5 = calc_ID(self.width,5)
        ID10 = calc_ID(self.width,10)
        
        plt.plot(ID5,num_saccade_D5,'ro-',label='Distance=5')
        plt.plot(ID10,num_saccade_D10,'bs--',label='Distance=10')

        plt.xlabel('Index of difficulty (bits) ',fontsize=13)
        plt.ylabel('Saccades per trial',fontsize=13)
        plt.ylim([0.9, 4])
        plt.title('Baseline Model',fontsize=13)
        plt.legend(fontsize=10)

        # To plot from data.
        plt.subplot(1,3,2)
        
        err = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.1, 0.1, 0.1, 0.1])
        mid = np.array([1.05, 1.08, 1.08, 1.08, 1.25, 1.30, 1.42, 1.55, 1.66])

        
        final_ID = np.array([1, 1.32192809, 1.73696559, 2, 2.32192809, 2.73696559, 3.32192809, 3.73696559, 4.32192809])

        plt.errorbar(final_ID, mid, yerr = err,color='k',fmt='s-',label='Pooled (D5 and D10)')
        plt.xlabel('Index of difficulty (bits) ',fontsize=14)
        plt.ylim([0.9, 4])
        plt.title('Data',fontsize=14)

        plt.legend(fontsize=10)
        
        # To plot from our model.
        plt.subplot(1,3,3)
        #Best Run Plot for 50Lakh timesteps
        saccade_mean_d5_50L = [2.06, 1.76, 1.38, 1.04, 1.0, 1.0] #one best run values
        saccade_mean_d10_50L = [2.66, 2.45, 1.95, 1.68, 1.3, 1.0] #one best run values
        plt.plot(ID5,saccade_mean_d5_50L,'ro-',label='Distance=5')
        plt.plot(ID10,saccade_mean_d10_50L,'bs--',label='Distance=10')
        
        
        plt.xlabel('Index of difficulty (bits) ',fontsize=13)
        plt.ylim([0.9, 4])
        plt.title('Custom Model',fontsize=14)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(self.fig_plot_path, 'fitts-saccade-ID.pdf'), bbox_inches='tight', dpi=300)

    # To plot the gaze jitters in target region.
    def jitter_average_radius(self):
        gaze_positions = np.array([[554, 459], [531, 465], [545, 463], [553, 475], [550, 421], [553, 464], [556, 450], [517, 435], [570, 453], [566, 446], [558, 480], [548, 450], [556, 449], [550, 448], [559, 450], [554, 451]]) # Your gaze position array
        target_center = np.array([0.02018172, 0.17336216])  # Center of the target circle
        target_width = 0.05726646259971647 #in radians
        screen_width = 1080
        screen_height = 720
        world_width = 2
        scale = screen_width / world_width
        reversed_gaze_positions = (gaze_positions - np.array([screen_width / 2, screen_height / 2])) / scale
        
        plt.figure(figsize=(8, 5))

        # Plot target circle
        circle = plt.Circle((target_center[0], target_center[1]), target_width, color='red', fill=True, label='Target')
        plt.gca().add_artist(circle)
        plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

        # Plot gaze positions over the target circle
        plt.scatter(reversed_gaze_positions[:, 0], reversed_gaze_positions[:, 1], c='black', marker='o', label='Gaze Positions')

        # Set plot limits and labels
        plt.xlim(reversed_gaze_positions[:, 0].min() - 0.1, reversed_gaze_positions[:, 0].max() + 0.1)
        plt.ylim(reversed_gaze_positions[:, 1].min() - 0.1, reversed_gaze_positions[:, 1].max() + 0.1)
        plt.title('Gaze Jitters over Target Circle')
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(self.fig_plot_path, 'jitter.pdf'), bbox_inches='tight', dpi=300)
        plt.grid()

    # To plot comparsion between MT and ID.
    def ID_MT_Saccades(self):
        # Data
        final_ID = np.array([1, 1.32192809, 1.73696559, 2, 2.32192809, 2.73696559, 3.32192809, 3.73696559, 4.32192809])
        final_saccades = np.array([1.0, 1.0, 1.0, 1.0, 1.34, 1.72, 2.0, 2.45, 2.66])
        final_MT = np.array([257.38666483, 257.38666483, 257.63402492, 310.72980496, 367.20306214, 443.20502614, 519.26474804, 604.46392766, 688.00895367])

        # Create a figure and axis
        plt.figure(figsize=(10, 5))

        # Plot final_saccades
        plt.subplot(1,2,1)
        plt.plot(final_ID, final_saccades, marker='o', linestyle='-', color='blue', label='Saccades Pooled (D5 and D10)')
        plt.xlabel('Index of difficulty (bits)',fontsize=13)
        plt.ylabel('Num. of saccades',fontsize=13)
        plt.title('Comparison of Num. of saccades and ID',fontsize=13)
        plt.legend(fontsize=10)

        # Plot final_MT
        plt.subplot(1,2,2)
        plt.plot(final_ID, final_MT, marker='o', linestyle='-', color='green', label='MT Pooled (D5 and D10)')

        # Add labels and title
        plt.xlabel('Index of difficulty (bits) ',fontsize=13)
        plt.ylabel('MT(ms)', fontsize=13)
        plt.title('Comparison of MT and ID',fontsize=13)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(self.fig_plot_path, 'IDMTSAC.pdf'), bbox_inches='tight', dpi=300)
        plt.show()


if __name__ == '__main__':
    obj = plot_extraction()
    #obj.jitter_average_radius() #Fig 3.5
    #obj.num_saccade() #Fig 4.1
    obj.sel_time() # Fig 4.2
    #obj.ID() #Fig 4.3
    #obj.ID_MT_Saccades() #Fig 4.4
    #obj.sel_time_jitter() # Fig 4.5



