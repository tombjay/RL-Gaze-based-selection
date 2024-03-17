# Gaze-based Selection
This repository contains the code for the development of an adaptive model for gaze-based selection using RL. To get to know our project more clearly,  kindly have a look at our ['report'](https://git.hcics.simtech.uni-stuttgart.de/theses/infotech2023_jayakumar/src/branch/main/Final-report.pdf/)

## Getting started
- Clone this repository and install the necessary packages listed in ['rl-environment.yml'](https://git.hcics.simtech.uni-stuttgart.de/theses/infotech2023_jayakumar/src/branch/main/rl-environment.yml) file by running the command,

```
conda env create -f rl-environment.yml
```
## How to train your model
### Step 1 - Custom environment
I used OpenAI gymnasium API to create our custom environment. This repository contains two custom environments: 
- GazeEnv.py - Custom environment that mimics gaze-based selection task without jitter adaptation.
- Jitter-GazeEnv.py - A custom environment that mimics gaze-based selection task with jitter adaptations.

These two environments serve the purpose of training and testing the model, as well as generating the required plots and graphs. 
In addition to these two '.py' files, there are also two '.ipynb' files that prove helpful for training and interactively testing the model.

### Step 2 - Training the model
Before training the model, I need to set the parameters in ['parameters.py'](https://git.hcics.simtech.uni-stuttgart.de/theses/infotech2023_jayakumar/src/branch/main/gazeselection/parameters.py) file. 
Once set, run the following code,

```
python main.py --cross_validation train --model_name core_PPO --train_mode static
```
By default training of the model would be started with a fixed combination of width and distance. To randomly initialise the combination, change the args from '--train_mode static' to '--train_mode random_init'. Once trained, the trained model will be saved as a zip file. I train our models with StableBaseline3's PPO algorithm.

To test the trained model, load the zip file path of the trained model in the 'main.py' file and run the following command,

```
python main.py --cross_validation test
```
I also used a 'Monitor' class to wrap our environment during training and testing. It records various information during the interaction with the environment, such as episode statistics, rewards, actions, and more. Information about the log path of the 'Monitor' class is provided inside the 'main.py' file.

## Evaluation & Results
The model's evaluation involved a comprehensive comparison of parameters, including target selection time and the count of saccades, against the baseline model's performance as documented by Chen et al. 2021. Furthermore, a key innovation of our work is the development of an adaptive model that accounts for fixational eye movements, or jitters. This jitter adaptive model stands out as a significant contribution to this project. To evaluate the jitter modelling, I have made some adjustments that are listed as in-code comments in the jitter environment files. 

Plots of these comparisons are made available inside the ['figures'](https://git.hcics.simtech.uni-stuttgart.de/theses/infotech2023_jayakumar/src/branch/main/figures/) folder.

