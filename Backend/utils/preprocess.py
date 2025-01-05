import requests
import ee

# API keys
PROJECT_ID = "landslideproject-446620"
OPENWEATHERMAP_API_KEY = "8c2a3c9245acd4626517a6d2d68d3ea8"

def initialize_gee():
    """
    Initialize Google Earth Engine with the specified project ID.
    """
    try:
        ee.Initialize(project=PROJECT_ID)
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

initialize_gee()

def get_lat_lon(location):
    """
    Get latitude and longitude for a location.
    """
    url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {'q': location, 'limit': 1, 'appid': OPENWEATHERMAP_API_KEY}
    response = requests.get(url, params=params).json()
    lat = round(response[0]['lat'], 1)
    lon = round(response[0]['lon'], 1)
    return lat, lon

def scale_to_model_range(value, min_value, max_value):
    """
    Scale value to range 1-5 and round to 1 decimal point.
    """
    if value < min_value:
        value = min_value
    elif value > max_value:
        value = max_value
    return round(1 + (value - min_value) * 4 / (max_value - min_value), 1)

def collect_and_preprocess_data(location):
    """
    Collect and preprocess real-time data for prediction.
    """
    lat, lon = get_lat_lon(location)

    # Precipitation
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {'lat': lat, 'lon': lon, 'appid': OPENWEATHERMAP_API_KEY, 'units': 'metric'}
    precipitation_data = requests.get(url, params=params).json()
    precipitation = precipitation_data.get('rain', {}).get('1h', 0)
    scaled_precipitation = scale_to_model_range(precipitation, 0, 50)

    # Terrain Features (Slope, Aspect, Elevation)
    aoi = ee.Geometry.Point([lon, lat]).buffer(5000)
    elevation = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    terrain = ee.Terrain.products(elevation)

    slope = terrain.select("slope").reduceRegion(ee.Reducer.mean(), aoi, 30).get("slope").getInfo()
    aspect = terrain.select("aspect").reduceRegion(ee.Reducer.mean(), aoi, 30).get("aspect").getInfo()
    elevation = elevation.reduceRegion(ee.Reducer.mean(), aoi, 30).get("elevation").getInfo()

    scaled_slope = scale_to_model_range(slope, 0, 60)
    scaled_aspect = scale_to_model_range(aspect, 0, 360)
    scaled_elevation = scale_to_model_range(elevation, 0, 8848)

    # NDVI and NDWI
    collection = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(aoi).sort("CLOUDY_PIXEL_PERCENTAGE").first()
    ndvi = collection.normalizedDifference(['B8', 'B4']).reduceRegion(ee.Reducer.mean(), aoi, 10).get("nd")
    ndwi = collection.normalizedDifference(['B8', 'B3']).reduceRegion(ee.Reducer.mean(), aoi, 10).get("nd")

    scaled_ndvi = scale_to_model_range(ndvi, -1, 1)
    scaled_ndwi = scale_to_model_range(ndwi, -1, 1)

    return [scaled_precipitation, scaled_slope, scaled_aspect, scaled_elevation, scaled_ndvi, scaled_ndwi]
