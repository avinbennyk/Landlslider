from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError, Field
from typing import Optional
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
class PredictionInput(BaseModel):
    location: Optional[str] = Field(None, example="Mumbai")
    latitude: Optional[float] = Field(None, example=19.0760)
    longitude: Optional[float] = Field(None, example=72.8777)

    @classmethod
    def validate(cls, values):
        """
        Validate that either location or latitude and longitude are provided,
        but not both simultaneously.
        """
        location = values.get("location")
        latitude = values.get("latitude")
        longitude = values.get("longitude")

        if location and (latitude is not None or longitude is not None):
            raise ValidationError("Provide either 'location' or both 'latitude' and 'longitude', but not both.")
        if not location and (latitude is None or longitude is None):
            raise ValidationError("Provide either 'location' or both 'latitude' and 'longitude'.")
        return values

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
def predict_landslide(data: PredictionInput):
    """
    Real-time landslide prediction endpoint.
    """
    try:
        # Step 1: Initialize Google Earth Engine
        initialize_gee()

        # Step 2: Fetch latitude and longitude for the provided location or use provided lat/lon
        if data.location:
            try:
                lat, lon = get_lat_lon_from_location(data.location)
            except Exception as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"Error fetching latitude and longitude for location '{data.location}': {e}. "
                           f"Please provide latitude and longitude manually."
                )
        else:
            lat, lon = data.latitude, data.longitude

        # Step 3: Gather real-time features
        precipitation = get_and_scale_precipitation(lat, lon)
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
                "message": f"Warning! A landslide is predicted for the location. Take immediate precaution.",
                "location": data.location or f"Latitude: {lat}, Longitude: {lon}"
            }
            alert_storage.append(alert_message)  # Store the alert in memory
            print(f"Generated alert: {alert_message}")  # Debugging log

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")

@router.get("/alerts", response_model=list[AlertOutput])
def get_alerts():
    """
    Retrieve all generated alerts.
    """
    return alert_storage
