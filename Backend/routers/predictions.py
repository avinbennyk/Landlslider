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
from utils.model_loader import predict  # Replace with your model prediction code

router = APIRouter()

# Input schema
class LocationInput(BaseModel):
    location: str  # Location name (e.g., "Mumbai")

# Output schema
class PredictionOutput(BaseModel):
    model: str
    prediction: str
    confidence: float

@router.post("/", response_model=PredictionOutput)
def predict_landslide(data: LocationInput):
    """
    Real-time landslide prediction endpoint.
    """
    try:
        # Ensure GEE is initialized
        initialize_gee()

        # Step 1: Get latitude and longitude for the location
        lat, lon = get_lat_lon_from_location(data.location, api_key="8c2a3c9245acd4626517a6d2d68d3ea8")

        # Step 2: Collect all real-time features
        precipitation = get_and_scale_precipitation(lat, lon, api_key="8c2a3c9245acd4626517a6d2d68d3ea8")
        terrain = compute_slope_aspect(lat, lon)
        curvature = compute_curvature(lat, lon)
        elevation = compute_elevation(lat, lon)
        vegetation = compute_ndvi_ndwi(lat, lon)

        # Combine all features into a single input vector
        features = [
            precipitation,
            terrain["scaled_slope"],
            terrain["scaled_aspect"],
            curvature,
            elevation,
            vegetation["scaled_ndvi"],
            vegetation["scaled_ndwi"]
        ]

        # Step 3: Pass the features to the model for prediction
        result = predict(features)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")
