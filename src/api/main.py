import numpy as np
import pandas as pd
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from src.config import MODEL_SAVE_PATH
from src.components.feature_engineering import create_features
from src.api.schemas import TaxiInput, PredictionOutput
from src.utils.logger import get_logger

logger = get_logger("api_service")
app = FastAPI(title="NYC Taxi Duration API", version="1.0")

# INSTALL THE MODEL GLOBALLY
try:
    logger.info(f"LOADING MODEL FROM {MODEL_SAVE_PATH}")
    sess = rt.InferenceSession(MODEL_SAVE_PATH)
    input_name = sess.get_inputs()[0].name
    logger.info("MODEL HAS SUCCESSFULLY LOADED")
except Exception as e:
    logger.error(f"MODEL HAS NOT LOADED: {e}")
    raise e

@app.get("/health")
def health_check():
    """CHECKS IF THE API IS ALIVE."""
    return {"status": "active", "model": "loaded"}

@app.get("/")
def root():
    return {"message": "WELCOME TO NYC TAXI DURATION API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: TaxiInput):
    try:
        # 1. CONVERT INCOMING JSON DATA TO DATAFRAME
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # 2. FEATURE ENGINEERING
        df_processed = create_features(df)

        # 3. SELECT THE COLUMNS THE MODEL EXCEPTS
        features = [
            'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'month', 'day_of_week', 'hour', 'is_weekend',
            'distance_haversine', 'distance_manhattan', 'bearing'
        ]

        # CONVERT THE DATAFRAME TO A FORMAT THE MODEL CAN UNDERSTAND (FLOAT32 ARRAY)
        X_input = df_processed[features].astype(np.float32).to_numpy()

        # 4. INFERENCE
        pred_log = sess.run(None, {input_name: X_input})[0]

        # 5. INVERSE LOG
        pred_seconds = np.expm1(pred_log)[0][0]

        logger.info(f"ESTIMATE MADE: {pred_seconds:.2f} SECONDS")

        return {
            "predicted_duration_seconds": round(float(pred_seconds), 2),
            "predicted_duration_minutes": round(float(pred_seconds / 60), 2)
        }

    except Exception as e:
        logger.error(f"PREDICTION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))