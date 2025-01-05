def send_alert_to_users(message):
    """
    Simulate sending an alert notification.
    """
    print(f"ALERT SENT: {message}")
    return {"status": "success", "message_sent": message}
