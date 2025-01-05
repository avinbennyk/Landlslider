import requests
import ee

# Replace with your project ID and OpenWeatherMap API key
PROJECT_ID = "landslideproject-446620"
OPENWEATHERMAP_API_KEY = "8c2a3c9245acd4626517a6d2d68d3ea8"

def initialize_gee():
    """
    Initialize Google Earth Engine with the specified project ID.
    """
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"Google Earth Engine initialized successfully with project: {PROJECT_ID}")
    except ee.EEException as e:
        print("Error initializing Google Earth Engine:", e)
        print("Re-authenticating...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

def get_lat_lon_from_location(location, api_key):
    """
    Get the latitude and longitude of a location using OpenWeatherMap Geocoding API.
    """
    geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': location,
        'limit': 1,
        'appid': api_key
    }
    try:
        response = requests.get(geocoding_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"Error: Location '{location}' not found.")
            raise ValueError("Manual Input Required")

        lat = data[0]['lat']
        lon = data[0]['lon']
        print(f"Location '{location}' found: Latitude={lat}, Longitude={lon}")
        return lat, lon
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching lat/lon for location '{location}': {e}")
        print("Please enter latitude and longitude manually.")
        lat = float(input("Enter Latitude: "))
        lon = float(input("Enter Longitude: "))
        return lat, lon

def get_and_scale_precipitation(lat, lon, api_key, min_value=0, max_value=50):
    """
    Fetch real-time precipitation data using Current Weather Data API
    and scale it to a range of 1 to 5.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        precipitation = data.get('rain', {}).get('1h', 0)  # Rainfall in the last hour
        print(f"Precipitation (last hour): {precipitation} mm")
        scaled_precipitation = scale_to_model_range(precipitation, min_value, max_value)
        return scaled_precipitation
    except requests.exceptions.RequestException as e:
        print(f"Error fetching precipitation data: {e}")
        return None

def scale_to_model_range(value, min_value, max_value):
    """
    Scale a feature to the range of 1 to 5.
    """
    if value < min_value:
        value = min_value
    elif value > max_value:
        value = max_value
    return round(1 + (value - min_value) * (5 - 1) / (max_value - min_value), 2)

def compute_slope_aspect(lat, lon, radius=5000):
    """
    Compute slope and aspect for the specified location and scale them to a range of 1 to 5.
    """
    aoi = ee.Geometry.Point([lon, lat]).buffer(radius)
    elevation = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    terrain = ee.Terrain.products(elevation)
    slope = terrain.select("slope")
    aspect = terrain.select("aspect")

    scaled_slope = slope.unitScale(0, 60).multiply(4).add(1).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e6
    ).get("slope")

    scaled_aspect = aspect.unitScale(0, 360).multiply(4).add(1).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e6
    ).get("aspect")

    return {
        "scaled_slope": round(scaled_slope.getInfo(), 2),
        "scaled_aspect": round(scaled_aspect.getInfo(), 2)
    }

def compute_curvature(lat, lon, radius=5000):
    """
    Compute curvature for the specified location and scale it to a range of 1 to 5.
    """
    aoi = ee.Geometry.Point([lon, lat]).buffer(radius)
    elevation = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    curvature = elevation.convolve(ee.Kernel.laplacian8())

    # Print raw curvature for debugging
    raw_curvature = curvature.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e6
    ).get("elevation")

    print(f"Raw Curvature Value: {raw_curvature.getInfo()}")  # Debugging

    scaled_curvature = curvature.unitScale(-10, 10).multiply(4).add(1).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e6
    ).get("elevation")

    return round(scaled_curvature.getInfo(), 2)

def compute_elevation(lat, lon, radius=5000):
    """
    Compute elevation for the specified location and scale it to a range of 1 to 5.
    """
    aoi = ee.Geometry.Point([lon, lat]).buffer(radius)
    elevation = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    scaled_elevation = elevation.unitScale(0, 8848).multiply(4).add(1).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e6
    ).get("elevation")

    return round(scaled_elevation.getInfo(), 2)

def compute_ndvi_ndwi(lat, lon, radius=5000):
    """
    Compute NDVI and NDWI for the specified location and scale them to a range of 1 to 5.
    """
    aoi = ee.Geometry.Point([lon, lat]).buffer(radius)
    collection = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterBounds(aoi) \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first()
    
    ndvi = collection.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = collection.normalizedDifference(['B8', 'B3']).rename('NDWI')

    scaled_ndvi = ndvi.unitScale(-1, 1).multiply(4).add(1).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e6
    ).get("NDVI")

    scaled_ndwi = ndwi.unitScale(-1, 1).multiply(4).add(1).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e6
    ).get("NDWI")

    return {
        "scaled_ndvi": round(scaled_ndvi.getInfo(), 2),
        "scaled_ndwi": round(scaled_ndwi.getInfo(), 2)
    }

if __name__ == "__main__":
    initialize_gee()

    location = input("Enter the location name: ")
    lat, lon = get_lat_lon_from_location(location, OPENWEATHERMAP_API_KEY)

    if lat is not None and lon is not None:
        precipitation = get_and_scale_precipitation(lat, lon, OPENWEATHERMAP_API_KEY)
        terrain_results = compute_slope_aspect(lat, lon)
        scaled_curvature = compute_curvature(lat, lon)
        vegetation_results = compute_ndvi_ndwi(lat, lon)
        scaled_elevation = compute_elevation(lat, lon)

        print(f"\nResults for {location}:")
        print(f"Scaled Precipitation: {precipitation}")
        print(f"Scaled Slope: {terrain_results['scaled_slope']}")
        print(f"Scaled Aspect: {terrain_results['scaled_aspect']}")
        print(f"Scaled Curvature: {scaled_curvature}")
        print(f"Scaled NDVI: {vegetation_results['scaled_ndvi']}")
        print(f"Scaled NDWI: {vegetation_results['scaled_ndwi']}")
        print(f"Scaled Elevation: {scaled_elevation}")
