from fastapi import APIRouter
from utils.alerts import send_alert_to_users

router = APIRouter()

@router.post("/alert")
def send_alert(message: str):
    """
    Send alert notifications to civilians in case of high landslide risk.
    """
    response = send_alert_to_users(message)
    return response
