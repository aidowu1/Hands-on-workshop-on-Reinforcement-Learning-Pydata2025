from datetime import datetime
import os

# Project name
PROJECT_ROOT_PATH = "grid-world"

# Settings configurations
SETTINGS_FOLDER = ""
DEFAULT_SETTINGS_NAME = ""

# Data file paths
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Model file paths
MODEL_FOLDER = "model"
os.makedirs(MODEL_FOLDER, exist_ok=True)
GRIDWORLD_QL_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER, "gridworld_q_learning_model.pl")


# LOGGING Configurations
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - : %(message)s in %(pathname)s:%(lineno)d'
CURRENT_DATE = datetime.today().strftime("%d%b%Y")
LOG_FILE = f"RL_Demo_Pydata2025_{CURRENT_DATE}.log"
LOG_FOLDER = "logs"
LOG_PATH = os.path.join(LOG_FOLDER, LOG_FILE)

# Miscellaneous settings
NEW_LINE = "\n"
LINE_DIVIDER = "==========" * 5

# Grid world configs
RENDER_MODE = "human"
GRID_SIZE = 20
IS_USE_FIXED_START_AND_PIT_POSITIONS = True
N_GRID_PITS = 3
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)