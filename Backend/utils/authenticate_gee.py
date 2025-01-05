import ee

# Replace with your Google Cloud project ID
PROJECT_ID = "landslideproject-446620"

try:
    # Authenticate with Earth Engine
    print("Authenticating with Google Earth Engine...")
    ee.Authenticate()

    # Initialize Earth Engine with your project ID
    print(f"Initializing Google Earth Engine for project: {PROJECT_ID}...")
    ee.Initialize(project=PROJECT_ID)

    print("Google Earth Engine authenticated and initialized successfully!")

except ee.EEException as e:
    print(f"Error during Earth Engine authentication or initialization: {e}")
    print("Ensure that your project is correctly set up with the Earth Engine API enabled.")
    print("Visit: https://developers.google.com/earth-engine/guides/access")
