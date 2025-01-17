from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.preprocess import (
    initialize_gee,
    get_lat_lon_from_location,
    get_and_scale_precipitation,
    compute_slope_aspect,
    compute_curvature,
    compute_elevation,
    compute_ndvi_ndwi
)
from utils.model_loader import predict  # Model loader and prediction function

router = APIRouter()

# In-memory storage for alerts
alert_storage = []

# Input schema
class LocationInput(BaseModel):
    location: str  # Name of the location (e.g., "Mumbai")

# Output schema for prediction
class PredictionOutput(BaseModel):
    model: str
    prediction: str
    confidence: float

# Output schema for alerts
class AlertOutput(BaseModel):
    message: str
    location: str

@router.post("/", response_model=PredictionOutput)
def predict_landslide(data: LocationInput):
    """
    Real-time landslide prediction endpoint.
    """
    try:
        # Step 1: Initialize Google Earth Engine
        initialize_gee()

        # Step 2: Fetch latitude and longitude for the provided location
        lat, lon = get_lat_lon_from_location(data.location, api_key="8c2a3c9245acd4626517a6d2d68d3ea8")

        # Step 3: Gather real-time features
        precipitation = get_and_scale_precipitation(lat, lon, api_key="8c2a3c9245acd4626517a6d2d68d3ea8")
        terrain = compute_slope_aspect(lat, lon)
        curvature = compute_curvature(lat, lon)
        elevation = compute_elevation(lat, lon)
        vegetation = compute_ndvi_ndwi(lat, lon)

        # Combine all features into a single vector
        features = [
            precipitation,
            terrain["scaled_slope"],
            terrain["scaled_aspect"],
            curvature,
            elevation,
            vegetation["scaled_ndvi"],
            vegetation["scaled_ndwi"]
        ]

        # Step 4: Perform prediction
        result = predict(features)

        # Step 5: Generate alert if a landslide is predicted
        if result["prediction"] == "Landslide":
            alert_message = {
                "message": f"Warning! A landslide is predicted for {data.location}. Take immediate precaution.",
                "location": data.location
            }
            alert_storage.append(alert_message)  # Store the alert in memory
            print(f"Generated alert: {alert_message}")  # Debugging log

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")

@router.get("/alerts", response_model=list[AlertOutput])
def get_alerts():
    """
    Retrieve all generated alerts.
    """
    return alert_storage
