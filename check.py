import pandas as pd
import joblib

# Load the trained model
model_path = '/Users/avinbennyk/Desktop/Landslidepro/Model/best_stacking_model.pkl'
model = joblib.load(model_path)

# Load the transformers
poly_path = '/Users/avinbennyk/Desktop/Landslidepro/Transformers/poly_transformer.pkl'
scaler_path = '/Users/avinbennyk/Desktop/Landslidepro/Transformers/scaler_transformer.pkl'

poly_transformer = joblib.load(poly_path)
scaler_transformer = joblib.load(scaler_path)

# Example feature values
example_features = {
    'Slope': 3.0, 'Aspect': 1.0, 'Curvature': 3.0, 'Precipitation': 0.5,
    'NDVI': 0.4, 'NDWI': 0.2, 'Elevation': 1.0
}

# Convert dictionary to DataFrame for model input
feature_df = pd.DataFrame([example_features])

# Apply polynomial features transformation and scaling
feature_poly = poly_transformer.transform(feature_df)
feature_scaled = scaler_transformer.transform(feature_poly)

# Make a prediction using the trained model
try:
    prediction = model.predict(feature_scaled)
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Error during model prediction: {e}")
