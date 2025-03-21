from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import Optional
from utils.preprocess import (
    initialize_gee,
    get_lat_lon_from_location,
    get_and_scale_precipitation,
    compute_slope_aspect,
    compute_curvature,
    compute_elevation,
    compute_ndvi_ndwi,
    get_weather_and_air_quality_details
)
from utils.model_loader import predict
import pandas as pd
import joblib
import logging
import pywhatkit as kit
import time

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load transformers
poly_path = "/Users/avinbennyk/Desktop/Landslidepro/Transformers/poly_transformer.pkl"
scaler_path = "/Users/avinbennyk/Desktop/Landslidepro/Transformers/scaler_transformer.pkl"

poly_transformer = joblib.load(poly_path)
scaler_transformer = joblib.load(scaler_path)

router = APIRouter()

# Hardcoded phone numbers for alerts (Replace with actual WhatsApp numbers)
RECEIVER_NUMBERS = [
    "+919876543210",  # Example number (replace with real numbers)
    "+918765432109"
]

class PredictionInput(BaseModel):
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @validator('latitude', 'longitude', always=True)
    def check_location_and_coordinates(cls, v, values, **kwargs):
        if 'location' in values and values['location'] is not None:
            if values.get('latitude') is not None or values.get('longitude') is not None:
                raise ValueError("Provide either 'location' or both 'latitude' and 'longitude', but not both.")
        elif 'location' not in values and (values.get('latitude') is None or values.get('longitude') is None):
            raise ValueError("Both 'latitude' and 'longitude' must be provided if 'location' is not specified.")
        return v

class PredictionOutput(BaseModel):
    model: str
    prediction: str
    confidence: float
    weather_details: dict

@router.post("/", response_model=PredictionOutput)
def predict_landslide(data: PredictionInput):
    try:
        logging.info(f"Received data for prediction: {data}")
        initialize_gee()

        lat, lon = None, None
        if data.location:
            lat, lon = get_lat_lon_from_location(data.location)
        elif data.latitude is not None and data.longitude is not None:
            lat, lon = data.latitude, data.longitude

        weather_details = get_weather_and_air_quality_details(lat, lon)

        precipitation = get_and_scale_precipitation(lat, lon)
        terrain = compute_slope_aspect(lat, lon)
        curvature = compute_curvature(lat, lon)
        elevation = compute_elevation(lat, lon)
        vegetation = compute_ndvi_ndwi(lat, lon)

        # Create feature dictionary
        features_dict = {
            'Slope': terrain['scaled_slope'],
            'Aspect': terrain['scaled_aspect'],
            'Curvature': curvature,
            'Precipitation': precipitation,
            'NDVI': vegetation['scaled_ndvi'],
            'NDWI': vegetation['scaled_ndwi'],
            'Elevation': elevation
        }

        # Convert dictionary to DataFrame
        feature_df = pd.DataFrame([features_dict])

        # Apply polynomial transformation and scaling
        feature_poly = poly_transformer.transform(feature_df)
        feature_scaled = scaler_transformer.transform(feature_poly)

        # Make a prediction using model_loader's predict function
        result = predict(feature_scaled)

        # Extract prediction result
        landslide_prediction = result["prediction"]
        confidence = result["confidence"]

                # If a landslide is predicted, send an alert
        if landslide_prediction.lower() == "landslide likely":
            alert_message = f"🚨 Landslide Alert! 🚨\nLocation: {data.location or f'Lat: {lat}, Lon: {lon}'}\nConfidence: {confidence:.2f}%\nTake necessary precautions! Stay safe."

            failed_numbers = []
            for number in RECEIVER_NUMBERS:
                try:
                    hour = time.localtime().tm_hour
                    minute = time.localtime().tm_min + 1  # Send in the next minute

                    kit.sendwhatmsg(number, alert_message, hour, minute, wait_time=10, tab_close=True)
                    logging.info(f"Alert sent to {number}")
                except Exception as e:
                    failed_numbers.append(number)
                    logging.error(f"Failed to send alert to {number}: {str(e)}")

            if failed_numbers:
                logging.warning(f"Failed to send alerts to: {', '.join(failed_numbers)}")

        return {
            'model': "StackingClassifier",
            'prediction': result["prediction"],  # Already formatted in model_loader
            'confidence': result["confidence"],
            'weather_details': weather_details
        }

    except Exception as e:
        logging.error(f"Error in processing prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
