from fastapi import FastAPI
from routers import predictions, admin

app = FastAPI(
    title="GeoPredict Backend",
    description="Real-time landslide prediction backend.",
    version="1.0.0"
)

# Include routers
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

@app.get("/")
def root():
    return {"message": "Welcome to GeoPredict Backend!"}
