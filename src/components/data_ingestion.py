import os

import pandas as pd

from src.config import NYC_BOUNDS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    logger.info(f"LOADING DATA FROM: {filepath}")

    if not os.path.exists(filepath):
        logger.error(f"FILE {filepath} DOES NOT EXIST")
        raise FileNotFoundError(f"FILE COULD NOT FIND: {filepath}")

    df = pd.read_csv(filepath)
    original_len = len(df)

    # 1. CLEANING: TIME
    df = df[(df["trip_duration"] >= 60) & (df["trip_duration"] <= 10800)]

    # 2. CLEANING: COORDINATE BOUNDARIES
    df = df[
        (df["pickup_longitude"] >= NYC_BOUNDS["min_lng"])
        & (df["pickup_longitude"] <= NYC_BOUNDS["max_lng"])
        & (df["pickup_latitude"] >= NYC_BOUNDS["min_lat"])
        & (df["pickup_latitude"] <= NYC_BOUNDS["max_lat"])
        & (df["dropoff_longitude"] >= NYC_BOUNDS["min_lng"])
        & (df["dropoff_longitude"] <= NYC_BOUNDS["max_lng"])
        & (df["dropoff_latitude"] >= NYC_BOUNDS["min_lat"])
        & (df["dropoff_latitude"] <= NYC_BOUNDS["max_lat"])
    ]

    logger.info(
        f"THE CLEANUP IS COMPLETE. THE REMAINING LINES ARE {original_len} -> {len(df)}"
    )
    return df
