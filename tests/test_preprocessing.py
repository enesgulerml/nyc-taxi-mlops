import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.components.feature_engineering import create_features


class TestFeatureEngineering:
    """
    Unit Tests for the Feature Engineering component.
    Tests whether the raw data can be transformed into the features (distance, hour, etc.) expected by the model.
    """

    @pytest.fixture
    def mock_raw_data(self):
        """
        It generates fake NYC Taxi data for testing purposes.
        """
        data = {
            # 1st Employee: Weekdays, 2nd Employee: Weekends
            "pickup_datetime": ["2026-01-20 10:00:00", "2026-01-24 10:00:00"],
            "dropoff_datetime": ["2026-01-20 10:15:00", "2026-01-24 10:20:00"],
            "passenger_count": [1, 2],
            "pickup_longitude": [-73.9857, -73.9857],
            "pickup_latitude": [40.7484, 40.7484],
            "dropoff_longitude": [-73.9665, -73.9665],
            "dropoff_latitude": [40.7812, 40.7812],
        }

        df = pd.DataFrame(data)

        # Datetime Transformation
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

        return df

    def test_distance_calculation(self, mock_raw_data):
        """
        Test: Does 'distance_haversine' column exist and is it logical?
        """
        processed_data = create_features(mock_raw_data)

        assert (
            "distance_haversine" in processed_data.columns
        ), "Important feature: 'distance_haversine' was not created."

        assert (
            processed_data["distance_haversine"].iloc[0] > 0
        ), "The distance was calculated as 0; there may be an error in the calculation."

    def test_time_features_structure(self, mock_raw_data):
        """
        Test: Are the time-based attributes (hour, day_of_week) being named correctly?
        """
        processed_data = create_features(mock_raw_data)

        required_features = ["hour", "day_of_week", "is_weekend"]

        for feature in required_features:
            assert feature in processed_data.columns, f"Feature '{feature}' is missing."

    def test_is_weekend_logic(self, mock_raw_data):
        """
        Test: Does the 'is_weekend' logic work correctly?
        """
        processed_data = create_features(mock_raw_data)

        # Row 1: Tuesday, January 20, 2026 -> Weekday (should be 0)
        assert (
            processed_data.iloc[0]["is_weekend"] == 0
        ), "Tuesday was marked as the weekend (ERROR)."

        # Row 2: Saturday, January 24, 2026 -> Weekend (should be 1)
        assert (
            processed_data.iloc[1]["is_weekend"] == 1
        ), "Saturday is marked as a weekday (ERROR)."

    def test_output_not_empty(self, mock_raw_data):
        """
        Test: The output should not be blank.
        """
        processed_data = create_features(mock_raw_data)
        assert not processed_data.empty, "The processed data returned nothing."
