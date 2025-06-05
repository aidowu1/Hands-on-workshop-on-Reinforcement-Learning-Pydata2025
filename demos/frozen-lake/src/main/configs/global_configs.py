from datetime import datetime
import os
import gymnasium as gym

# Project name
PROJECT_ROOT_PATH = "frozen-lake"

# Data file paths
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Model file paths
MODEL_FOLDER = "model"
os.makedirs(MODEL_FOLDER, exist_ok=True)
FROZEN_LAKE_QL_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER, "frozen_lake_q_learning_model.pl")
FROZEN_LAKE_SARSA_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER, "frozen_lake_sarsa_model.pl")
FROZEN_LAKE_DQN_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER, "frozen_lake_dqn_model.pl")


# LOGGING Configurations
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - : %(message)s in %(pathname)s:%(lineno)d'
CURRENT_DATE = datetime.today().strftime("%d%b%Y")
LOG_FILE = f"RL_Demo_Pydata2025_{CURRENT_DATE}.log"
LOG_FOLDER = "logs"
LOG_PATH = os.path.join(LOG_FOLDER, LOG_FILE)

# Miscellaneous settings
NEW_LINE = "\n"
LINE_DIVIDER = "==========" * 5

# Frozen-lake environment configurations
RENDER_MODE = "rgb_array"
FROZEN_LAKE_ENV: gym.Env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=RENDER_MODE)
SEED = 100
EPISODE_UPDATE_FREQUENCY = 1000

# Rendering configurations
DPI = 72
INTERVAL = 100 # ms

# Q-Learning configurations
Q_LEARN_ALPHA = 0.1
Q_LEARN_GAMMA = 0.99
Q_LEARN_EPSILON = 1.0
Q_LEARN_EPSILON_DECAY = 0.995
Q_LEARN_MIN_EPSILON = 0.01
Q_LEARN_N_EPISODES = 10000
Q_LEARN_MAX_STEPS = 100



