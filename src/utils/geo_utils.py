import numpy as np


def haversine_array(lat1, lng1, lat2, lng2):
    """Vector distance calculation (km) for Haversine."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # km
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    d = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    """Simplified Manhattan distance."""
    a = haversine_array(lat1, lng1, lat2, lng2)
    b = haversine_array(lat1, lng1, lat2, lng2)
    return a + b


def calculate_bearing(lat1, lng1, lat2, lng2):
    """It calculates the direction (angle) between two points."""
    dLon = np.radians(lng2 - lng1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return np.degrees(np.arctan2(y, x))
