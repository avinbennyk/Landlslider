from fastapi import APIRouter
from pydantic import BaseModel
from utils.preprocess import collect_and_preprocess_data
from utils.model_loader import predict

router = APIRouter()

# Input schema
class LocationInput(BaseModel):
    location: str  # Location name (e.g., "Kathmandu")

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
    # Step 1: Collect and preprocess real-time data
    features = collect_and_preprocess_data(data.location)

    # Step 2: Run predictions using the model
    result = predict(features)

    return result
