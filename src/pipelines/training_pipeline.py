import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Project Modules
from src.config import DATA_RAW_PATH, MODEL_SAVE_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.components.data_ingestion import load_and_clean_data
from src.components.feature_engineering import create_features
from src.utils.logger import get_logger

logger = get_logger("training_pipeline")


def run_training():
    """
    Executes the training pipeline with hyperparameter tuning simulation.
    Logs all experiments to MLflow and saves the best model to disk.
    """
    try:
        logger.info("🚀 TRAINING PIPELINE INITIALIZED")

        # 1. MLFLOW SETUP
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"📡 MLFLOW TRACKING URI: {MLFLOW_TRACKING_URI}")

        # 2. DATA LOADING & PREPROCESSING
        logger.info("💾 LOADING AND CLEANING RAW DATA...")
        df = load_and_clean_data(DATA_RAW_PATH)

        logger.info("🛠️ APPLYING FEATURE ENGINEERING...")
        df_processed = create_features(df)

        # Velocity Filter (Domain Logic)
        df_processed['avg_speed_kph'] = (df_processed['distance_haversine'] / df_processed['trip_duration']) * 3600
        df_processed = df_processed[(df_processed['avg_speed_kph'] <= 100) & (df_processed['avg_speed_kph'] >= 0.1)]

        df_processed['trip_duration_log'] = np.log1p(df_processed['trip_duration'])

        # Prepare Features and Target
        features = [
            'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'month', 'day_of_week', 'hour', 'is_weekend',
            'distance_haversine', 'distance_manhattan', 'bearing'
        ]
        target = 'trip_duration_log'

        # Validation: Check if columns exist
        if target not in df_processed.columns:
            raise KeyError(f"Target column '{target}' not found in DataFrame!")

        X = df_processed[features]
        y = df_processed[target]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define simulation parameters
        NUM_TRIALS = 30
        best_rmse = float('inf')
        best_model = None
        best_params = {}

        logger.info(f"🧪 Starting Hyperparameter Tuning ({NUM_TRIALS} Trials)...")

        # 3. HYPERPARAMETER TUNING LOOP
        for i in range(1, NUM_TRIALS + 1):

            # Randomly sample hyperparameters
            params = {
                "n_estimators": np.random.randint(50, 150),
                "max_depth": np.random.randint(5, 20),
                "min_samples_split": np.random.randint(2, 10),
                "min_samples_leaf": np.random.randint(1, 5),
                "random_state": 42
            }

            with mlflow.start_run(run_name=f"Trial_{i:02d}"):
                # TRAIN
                model = RandomForestRegressor(**params, n_jobs=-1)
                model.fit(X_train, y_train)

                # PREDICT & EVALUATE
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                # LOG TO MLFLOW
                mlflow.log_params(params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                logger.info(f"Trial {i}/{NUM_TRIALS} | ROOT MSE: {rmse:.4f} | Params: {params}")

                # Check if this is the best model so far
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = params
                    logger.info(f"🌟 NEW BEST MODEL FOUND! (ROOT MSE: {best_rmse:.4f})")

                    # Tag this run as candidate
                    mlflow.set_tag("candidate", "true")

        # 4. SAVE THE BEST MODEL (ONNX Export)
        if best_model:
            logger.info(f"📦 EXPORTING THE BEST MODEL (ROOT MSE: {best_rmse:.4f}) TO ONNX...")

            # Convert to ONNX
            initial_type = [('float_input', FloatTensorType([None, len(features)]))]
            onnx_model = convert_sklearn(best_model, initial_types=initial_type)

            # Save to disk
            with open(MODEL_SAVE_PATH, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"✅ BEST MODEL SAVED SUCCESSFULLY TO: {MODEL_SAVE_PATH}")

            # Log Best Model details to MLflow explicitly
            with mlflow.start_run(run_name="Best_Model_Final"):
                mlflow.log_params(best_params)
                mlflow.log_metric("final_best_rmse", best_rmse)
                mlflow.sklearn.log_model(best_model, "best_random_forest_model")
                logger.info("🏆 BEST MODEL ARTIFACTS UPLOADED TO MLFLOW REGISTRY.")

        logger.info("🏁 TRAINING PIPELINE SUCCESSFULLY COMPLETED")

    except Exception as e:
        logger.error(f"❌ CRITICAL FAILURE IN PIPELINE EXECUTION: {e}")
        raise e


if __name__ == "__main__":
    run_training()