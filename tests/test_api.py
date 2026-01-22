import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


@patch("src.api.main.model")
@patch("src.api.main.cache")
def test_predict_endpoint(mock_cache, mock_model):
    # 1. Cache Miss (Empty)
    mock_cache.get.return_value = None

    # 2. Model Mocking
    # The ONNX Runtime .run() method returns a LIST.
    # The first element of the list is a Numpy Array.
    # We are mimicking the same thing:
    mock_output = np.array([[2.7]])  # Log(15 seconds) approximate value
    mock_model.run.return_value = [mock_output]

    payload = {
        "pickup_datetime": "2026-01-20 12:00:00",
        "dropoff_datetime": "2026-01-20 12:15:00",
        "passenger_count": 1,
        "pickup_longitude": -73.9857,
        "pickup_latitude": 40.7484,
        "dropoff_longitude": -73.9665,
        "dropoff_latitude": 40.7812,
    }

    response = client.post("/predict", json=payload)

    # For debugging: See the text if you get an error.
    assert response.status_code == 200, f"API Error: {response.text}"

    data = response.json()
    assert "predicted_duration_seconds" in data
    assert data["predicted_duration_seconds"] > 0


def test_predict_invalid_data():
    bad_payload = {"pickup_datetime": "2026-01-20 12:00:00"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
