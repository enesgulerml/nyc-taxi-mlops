import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import mean_squared_error, r2_score

from src.config import RANDOM_STATE, TEST_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate(df: pd.DataFrame):
    # FEATURE SELECTION
    features = [
        'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'month', 'day_of_week', 'hour', 'is_weekend',
        'distance_haversine', 'distance_manhattan', 'bearing'
    ]

    X = df[features]
    y = np.log1p(df['trip_duration'])

    logger.info(f"TRAINING BEGINS... DATA SIZE: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingRegressor(
            max_iter=300, max_depth=10, learning_rate=0.1, random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    logger.info(f"MODEL TRAINED. METRICS (LOG SCALE) -> ROOT MSE: {rmse:.4f}, R2: {r2:.4f}")

    return pipeline, len(features)


def export_model(pipeline, feature_count, path):
    logger.info(f"THE MODEL SAVED IN ONNX FORMAT: {path}")
    initial_type = [('float_input', FloatTensorType([None, feature_count]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("✅ THE MODEL HAS BEEN SUCCESSFULLY SAVED.")