from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predictions, admin

app = FastAPI(
    title="GeoPredict Backend",
    description="Real-time landslide prediction backend.",
    version="1.0.0"
)

# Set up CORS middleware
origins = [
    "http://localhost:3001",  # Adjust the port if your frontend runs on a different one
    "http://127.0.0.1:3000"   # You can include more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

@app.get("/")
def root():
    return {"message": "Welcome to GeoPredict Backend!"}
