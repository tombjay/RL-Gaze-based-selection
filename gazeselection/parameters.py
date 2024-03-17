'''
This file contains the parameters for all the code.
Acts as a substitute for 'GIN-Config'.
'''
########################################Main-file#####################################
n_episodes = 500
ocular_noise = 0.07
spatial_noise = 0.09
width = [1, 1.5, 2, 3, 4, 5]
distance = [5, 10]
timesteps = 300000
log_path = r"D:\Research-Project\infotech2023_jayakumar\log-directory"
n_episodes = 5 #for testing the model.
num_epochs = 2
batch_size = 2

######################################Environment-files###############################
max_fixations = 10
target_onset_latency = 200 #ms

#After regression, the AR values for our target sizes in pairs:(AR, Target size)
#[(0.13, 1), (0.15, 1.5), (0.18, 2), (0.24, 3), (0.27, 4), (0.29, 5)] in degrees.
average_radius = 0.29