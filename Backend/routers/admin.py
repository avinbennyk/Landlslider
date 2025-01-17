from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter()

# Define the model for the alert message
class AlertMessage(BaseModel):
    message: str = Field(..., min_length=5, max_length=500, description="Alert message to be sent")
    location: Optional[str] = Field(None, description="Location where the alert is applicable")

# Define the model for retrieving alerts
class StoredAlert(BaseModel):
    message: str
    location: Optional[str]

# In-memory storage for the alert messages
alert_storage: List[StoredAlert] = []

@router.post("/alert", tags=["Admin"])
async def send_alert(alert: AlertMessage):
    """
    Endpoint to send alert messages to people.
    """
    if not alert.message.strip():
        raise HTTPException(status_code=400, detail="Alert message cannot be empty")

    # Add the alert to the in-memory storage
    new_alert = {"message": alert.message, "location": alert.location}
    alert_storage.append(new_alert)
    print(f"Alert sent: {alert.message} (Location: {alert.location})")  # Debug log

    return {
        "status": "success",
        "message": "Alert has been sent successfully",
        "alert": new_alert
    }

@router.get("/alerts", tags=["Admin"], response_model=List[StoredAlert])
async def get_all_alerts():
    """
    Endpoint to retrieve all sent alerts.
    """
    if not alert_storage:
        return {"message": "No alerts available at the moment"}
    return alert_storage
