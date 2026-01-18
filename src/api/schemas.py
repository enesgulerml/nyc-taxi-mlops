from pydantic import BaseModel
from datetime import datetime

# DATA COMING FROM THE USER
class TaxiInput(BaseModel):
    pickup_datetime: str  # "2024-01-18 14:30:00" | We are waiting in the format
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "pickup_datetime": "2026-01-18 14:30:00",
                "pickup_longitude": -73.985,
                "pickup_latitude": 40.758,
                "dropoff_longitude": -73.996,
                "dropoff_latitude": 40.732,
                "passenger_count": 1
            }
        }

class PredictionOutput(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float