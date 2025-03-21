import pywhatkit as kit
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import time

router = APIRouter()

# Hardcoded phone numbers (Replace with actual numbers in international format)
RECEIVER_NUMBERS = [
    "+919876543210",  # Example number (replace with real numbers)
    "+918765432109"
]

class AlertMessage(BaseModel):
    message: str = Field(..., min_length=5, max_length=500, description="Alert message to be sent")

@router.post("/alert", tags=["Admin"])
async def send_alert(alert: AlertMessage):
    """
    Send a WhatsApp alert message using PyWhatKit.
    """
    if not alert.message.strip():
        raise HTTPException(status_code=400, detail="Alert message cannot be empty")

    failed_numbers = []
    for number in RECEIVER_NUMBERS:
        try:
            # Send message immediately (hour, minute must be at least 1 min in the future)
            hour = time.localtime().tm_hour
            minute = time.localtime().tm_min + 1  # Send in the next minute

            kit.sendwhatmsg(number, alert.message, hour, minute, wait_time=10, tab_close=True)
            print(f"Alert sent to {number}")  # Debug log
        except Exception as e:
            failed_numbers.append(number)
            print(f"Failed to send alert to {number}: {str(e)}")

    if failed_numbers:
        raise HTTPException(status_code=500, detail=f"Failed to send alerts to: {', '.join(failed_numbers)}")

    return {"status": "success", "message": "Alerts sent successfully"}
