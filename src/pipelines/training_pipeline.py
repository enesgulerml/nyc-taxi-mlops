from src.config import DATA_RAW_PATH, MODEL_SAVE_PATH
from src.components.data_ingestion import load_and_clean_data
from src.components.feature_engineering import create_features
from src.components.model_trainer import train_and_evaluate, export_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training():
    try:
        logger.info("🚀 TRAINING PIPELINE LAUNCHED")

        # 1. LOAD
        df = load_and_clean_data(DATA_RAW_PATH)

        # 2. FEATURE ENGINEERING
        logger.info("FEATURE ENGINEERING IS BEING IMPLEMENTED...")
        df_processed = create_features(df)

        # WE CAN IMPLEMENT THE VELOCITY FILTER AT THE PIPELINE LEVEL (OPTIONAL)
        df_processed['avg_speed_kph'] = (df_processed['distance_haversine'] / df_processed['trip_duration']) * 3600
        df_processed = df_processed[(df_processed['avg_speed_kph'] <= 100) & (df_processed['avg_speed_kph'] >= 0.1)]

        # 3. TRAIN
        pipeline, n_features = train_and_evaluate(df_processed)

        # 4. SAVE
        export_model(pipeline, n_features, MODEL_SAVE_PATH)

        logger.info("🏁 TRAINING PIPELINE SUCCESSFULLY COMPLETED")

    except Exception as e:
        logger.error(f"PIPELINE STOPPED DUE TO ERROR: {e}")
        raise e


if __name__ == "__main__":
    run_training()