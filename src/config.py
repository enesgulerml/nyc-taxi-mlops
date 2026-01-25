import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- CI/CD & DATA SELECTION LOGIC ---
if os.getenv("CI") == "true":
    DATA_FILENAME = "sample_data.csv"
else:
    DATA_FILENAME = "train.csv"

# DATA PATHS
DATA_RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", DATA_FILENAME)
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "nyc_taxi_model.onnx")
LOG_FILE_PATH = os.path.join(ROOT_DIR, "logs", "running_logs.log")

# MODEL PARAMETERS
NYC_BOUNDS = {
    "min_lng": -74.3,
    "max_lng": -73.7,
    "min_lat": 40.5,
    "max_lat": 40.9,
}

# TRAINING SETTINGS
RANDOM_STATE = 42
TEST_SIZE = 0.2

# MLFLOW CONFIG
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "NYC_Taxi_V1"