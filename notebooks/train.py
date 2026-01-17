import pandas as pd
import numpy as np
import time
import os
import warnings
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

warnings.filterwarnings("ignore")

# --- SETTINGS ---
DATA_PATH = os.path.join("../data", "raw", "train.csv")
MODEL_PATH = "../nyc_taxi_model.onnx"

# NYC BORDERS (FOR OUTLIER CLEANUP)
NYC_BOUNDS = {
    'min_lng': -74.3, 'max_lng': -73.7,
    'min_lat': 40.5, 'max_lat': 40.9
}


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # km
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    d = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def calculate_bearing(lat1, lng1, lat2, lng2):
    dLon = np.radians(lng2 - lng1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return np.degrees(np.arctan2(y, x))


def load_and_prep_data(filepath):
    print(f"--- [1/4] LOADING AND CLEANING DATA: {filepath} ---")
    if not os.path.exists(filepath):
        print(f"ERROR: '{filepath}' NOT FOUND.")
        exit(1)

    df = pd.read_csv(filepath)
    original_len = len(df)

    # --- 1. DATA CLEANING ---
    df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 10800)]

    # COORDINATE FILTERING (REMOVE THOSE OUTSIDE NYC)
    df = df[
        (df['pickup_longitude'] >= NYC_BOUNDS['min_lng']) & (df['pickup_longitude'] <= NYC_BOUNDS['max_lng']) &
        (df['pickup_latitude'] >= NYC_BOUNDS['min_lat']) & (df['pickup_latitude'] <= NYC_BOUNDS['max_lat']) &
        (df['dropoff_longitude'] >= NYC_BOUNDS['min_lng']) & (df['dropoff_longitude'] <= NYC_BOUNDS['max_lng']) &
        (df['dropoff_latitude'] >= NYC_BOUNDS['min_lat']) & (df['dropoff_latitude'] <= NYC_BOUNDS['max_lat'])
        ]

    print(f"CLEANUP RESULT: {original_len} -> {len(df)} ROW (REMAINING: % {len(df) / original_len * 100:.1f})")

    print("--- [2/4] FEATURE ENGINEERING ---")
    start_time = time.time()

    # HISTORICAL DATA
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['month'] = df['pickup_datetime'].dt.month
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour

    # NEW FEATURE: WEEKEND?
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # DISTANCES
    df['distance_haversine'] = haversine_array(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    df['distance_manhattan'] = dummy_manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # SPEED CONTROL: LET'S REMOVE TAXIS GOING FASTER THAN 100 KM/H (GPS ERROR).
    # trip_duration second, distance km -> (dist/time)*3600 = km/h
    df['avg_speed_kph'] = (df['distance_haversine'] / df['trip_duration']) * 3600
    df = df[df['avg_speed_kph'] <= 100]
    df = df[df['avg_speed_kph'] >= 0.1]  # THROW AWAY THOSE THAT ARE NOT MOVING.

    # DIRECTION (BEARING)
    df['bearing'] = calculate_bearing(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # FEATURE SELECTION
    features = [
        'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'month', 'day_of_week', 'hour', 'is_weekend',
        'distance_haversine', 'distance_manhattan', 'bearing'
    ]

    X = df[features]
    y = df['trip_duration']

    # LOG TRANSFORM
    y = np.log1p(y)

    print(f"ENGINEERING TIME: {time.time() - start_time:.2f} SECONDS")
    return X, y, features


def train_model(X, y):
    print(f"--- [3/4] MODEL IS BEING TRAINED (WITH {len(X)} ROWS...) ---")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MODEL PARAMETERS
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modals', HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # --- DETAILED METRIC REPORT ---
    predictions_log = pipeline.predict(X_test)

    # LOG SCALE METRICS
    rmse_log = np.sqrt(mean_squared_error(y_test, predictions_log))
    r2_log = r2_score(y_test, predictions_log)

    # ORIGINAL SCALE METRICS
    y_test_orig = np.expm1(y_test)
    predictions_orig = np.expm1(predictions_log)

    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, predictions_orig))
    mae_orig = mean_absolute_error(y_test_orig, predictions_orig)
    r2_orig = r2_score(y_test_orig, predictions_orig)

    print("\n" + "=" * 40)
    print("        MODEL REPORT      ")
    print("=" * 40)
    print(f"LOG SCALE:")
    print(f"  > ROOT MSE : {rmse_log:.4f}")
    print(f"  > R2   : {r2_log:.4f}")
    print("-" * 40)
    print(f"ORIGINAL SCALE:")
    print(f"  > ROOT MSE : {rmse_orig:.2f} SECONDS")
    print(f"  > MAE  : {mae_orig:.2f} SECONDS (AVERAGE ERROR: {mae_orig / 60:.1f} MINUTES)")
    print(f"  > R2   : {r2_orig:.4f}")
    print("=" * 40 + "\n")

    return pipeline


def save_onnx(pipeline, feature_count):
    print("--- [4/4] ONNX EXPORT ---")

    initial_type = [('float_input', FloatTensorType([None, feature_count]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ SUCCESSFUL: MODEL SAVED AS '{MODEL_PATH}'")


if __name__ == "__main__":
    try:
        X, y, feature_names = load_and_prep_data(DATA_PATH)
        trained_pipeline = train_model(X, y)
        save_onnx(trained_pipeline, len(feature_names))
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")