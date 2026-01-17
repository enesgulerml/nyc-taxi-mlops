import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# DATA PATHS
DATA_RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "train.csv")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "modals", "nyc_taxi_model.onnx")
LOG_FILE_PATH = os.path.join(ROOT_DIR, "logs", "running_logs.log")

# MODEL PARAMETERS
NYC_BOUNDS = {
    'min_lng': -74.3, 'max_lng': -73.7,
    'min_lat': 40.5, 'max_lat': 40.9,
}

# TRAINING SETTINGS
RANDOM_STATE = 42
TEST_SIZE = 0.2