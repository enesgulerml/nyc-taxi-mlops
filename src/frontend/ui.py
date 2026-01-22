import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# API ADDRESS
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="NYC Taxi Time Prediction",
    layout="centered",
)

# TITLE AND DESCRIPTION
st.title("NYC Taxi Time Prediction")
st.markdown("""
This application uses an **ONNX model** trained with an MLOps pipeline and a **FastAPI** service. Please enter your journey details below.
""")

st.divider()

# INPUT FORM
col1, col2 = st.columns(2)

with col1:
    pickup_date = st.date_input("📅 Trip Date", datetime.now())
    pickup_time = st.time_input("⏰ Trip Time", datetime.now())
    passenger_count = st.number_input(
        "👤 Number Of Passengers", min_value=1, max_value=6, value=1
    )

with col2:
    pickup_lat = st.number_input(
        "📍 Acquisition Latitude (Lat.)", value=40.7580, format="%.4f"
    )
    pickup_lon = st.number_input(
        "📍 Acquisition Longitude (Lon)", value=-73.9855, format="%.4f"
    )
    dropoff_lat = st.number_input(
        "🏁 Arrival Latitude (Lat.)", value=40.7320, format="%.4f"
    )
    dropoff_lon = st.number_input(
        "🏁 Arrival Longitude (Lon)", value=-73.9960, format="%.4f"
    )

# PREDICTION BUTTON
st.divider()
if st.button("🚀 Estimate the time", type="primary", use_container_width=True):

    # 1. PREPARE DATA
    combined_datetime = f"{pickup_date} {pickup_time}"

    payload = {
        "pickup_datetime": combined_datetime,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "passenger_count": passenger_count,
    }

    # 2. SEND A REQUEST TO THE API
    with st.spinner("The model is communicating with the API..."):
        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                minutes = result["predicted_duration_minutes"]
                seconds = result["predicted_duration_seconds"]

                st.success("✅ Prediction Successful!")

                # METRIC NOTATION
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Estimated Time (Min)", f"{minutes} dk")
                m_col2.metric("In Seconds", f"{seconds:.1f} sn")

            else:
                st.error(f"Error: API returned code {response.status_code}.")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ Error: Could not connect to the API!")
            st.info(
                "💡 Hint: Are you sure the command 'uvicorn src.api.main:app' is running?"
            )
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


# SIDEBAR
with st.sidebar:
    st.header("ℹ️ About the Project")
    st.info("This project is an example of an End-to-End MLOps architecture.")
    st.write(f"**Backend:** FastAPI")
    st.write(f"**Frontend:** Streamlit")
    st.write(f"**Model:** Sklearn -> ONNX")
