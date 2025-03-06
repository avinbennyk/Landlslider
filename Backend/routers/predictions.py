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
    get_weather_and_air_quality_details  # Ensure this is correctly imported
)
from utils.model_loader import predict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# In-memory storage for alerts
alert_storage = []

# Input schema
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

# Output schema for prediction
class PredictionOutput(BaseModel):
    model: str
    prediction: str
    confidence: float
    weather_details: dict  # Including weather details in the output

# Output schema for alerts
class AlertOutput(BaseModel):
    message: str
    location: str

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

        weather_details =get_weather_and_air_quality_details(lat, lon)  # Fetching weather details

        precipitation = get_and_scale_precipitation(lat, lon)
        terrain = compute_slope_aspect(lat, lon)
        curvature = compute_curvature(lat, lon)
        elevation = compute_elevation(lat, lon)
        vegetation = compute_ndvi_ndwi(lat, lon)

        features = [
            precipitation,
            terrain['scaled_slope'],
            terrain['scaled_aspect'],
            curvature,
            elevation,
            vegetation['scaled_ndvi'],
            vegetation['scaled_ndwi']
        ]

        result = predict(features)

        if result['prediction'] == "Landslide":
            alert_message = {
                "message": f"Warning! A landslide is predicted for {data.location or f'Latitude: {lat}, Longitude: {lon}'}. Take immediate precaution.",
                "location": data.location or f"Latitude: {lat}, Longitude: {lon}"
            }
            alert_storage.append(alert_message)
            logging.info(f"Generated alert: {alert_message}")

        return {
            'model': result['model'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'weather_details': weather_details  # Adding weather details to the response
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in processing prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=list[AlertOutput])
def get_alerts():
    return alert_storage
