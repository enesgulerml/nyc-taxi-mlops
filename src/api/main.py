import numpy as np
import pandas as pd
import onnxruntime as rt
import os
import redis
import json
import hashlib
from fastapi import FastAPI, HTTPException
from src.config import MODEL_SAVE_PATH
from src.components.feature_engineering import create_features
from src.api.schemas import TaxiInput, PredictionOutput
from src.utils.logger import get_logger
from prometheus_fastapi_instrumentator import Instrumentator

# INITIALIZE LOGGER
logger = get_logger("api_service")

# INITIALIZE APP
app = FastAPI(title="NYC Taxi Duration API", version="2.0")

# MONITORING (PROMETHEUS)
Instrumentator().instrument(app).expose(app)

# --- REDIS CONFIGURATION ---
# "redis_cache" comes from docker-compose service name
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379

try:
    # Connect with a short timeout to avoid hanging if Redis is down
    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=1)
    cache.ping()  # Check connection
    logger.info(f"✅ CONNECTED TO REDIS AT {REDIS_HOST}:{REDIS_PORT}")
    redis_available = True
except redis.ConnectionError:
    logger.warning("⚠️ REDIS UNREACHABLE! CACHING DISABLED.")
    redis_available = False
# ---------------------------

# GLOBAL MODEL SESSION
sess = None
input_name = None


@app.on_event("startup")
def load_model():
    """LOADS THE MODEL ONCE WHEN API STARTS"""
    global sess, input_name
    try:
        logger.info(f"LOADING MODEL FROM {MODEL_SAVE_PATH}")
        sess = rt.InferenceSession(MODEL_SAVE_PATH)
        input_name = sess.get_inputs()[0].name
        logger.info("✅ MODEL HAS SUCCESSFULLY LOADED")
    except Exception as e:
        logger.error(f"❌ MODEL FAILED TO LOAD: {e}")
        raise e


def generate_cache_key(data: TaxiInput) -> str:
    """GENERATES A UNIQUE MD5 HASH FOR THE INPUT DATA"""
    # Convert input object to a sorted JSON string to ensure consistency
    data_str = json.dumps(data.dict(), sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


@app.get("/health")
def health_check():
    """CHECKS IF THE API AND REDIS ARE ALIVE."""
    return {
        "status": "active",
        "model": "loaded" if sess else "not_loaded",
        "redis": "connected" if redis_available else "disconnected"
    }


@app.get("/")
def root():
    return {"message": "WELCOME TO NYC TAXI DURATION API (WITH REDIS CACHE)"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: TaxiInput):
    try:
        # --- 1. CHECK CACHE (HIT) ---
        if redis_available:
            cache_key = generate_cache_key(data)
            cached_result = cache.get(cache_key)

            if cached_result:
                logger.info(f"⚡ CACHE HIT: {cache_key}")
                # Return stored JSON directly
                return json.loads(cached_result)

        # --- 2. CACHE MISS -> RUN INFERENCE ---

        # A. Convert Incoming JSON to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # B. Feature Engineering (Using the same pipeline component)
        df_processed = create_features(df)

        # C. Select Features Expected by Model
        features = [
            'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'month', 'day_of_week', 'hour', 'is_weekend',
            'distance_haversine', 'distance_manhattan', 'bearing'
        ]

        # D. Prepare Input (Float32)
        X_input = df_processed[features].astype(np.float32).to_numpy()

        # E. ONNX Inference
        pred_log = sess.run(None, {input_name: X_input})[0]

        # F. Inverse Log Transformation
        pred_seconds = np.expm1(pred_log)[0][0]

        # Prepare Response Object
        response_data = {
            "predicted_duration_seconds": round(float(pred_seconds), 2),
            "predicted_duration_minutes": round(float(pred_seconds / 60), 2)
        }

        logger.info(f"🧠 MODEL INFERENCE: {pred_seconds:.2f} SECONDS")

        # --- 3. SAVE TO CACHE (TTL: 1 HOUR) ---
        if redis_available:
            # Save the result to Redis for 3600 seconds
            cache.setex(cache_key, 3600, json.dumps(response_data))

        return response_data

    except Exception as e:
        logger.error(f"❌ PREDICTION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))