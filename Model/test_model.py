import joblib
import numpy as np

# Load the model
model_path = "/Users/avinbennyk/Desktop/Landslidepro/Model/best_stacking_model.pkl"
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()

# Test with dummy input features (replace with actual feature size)
# Ensure the feature count matches the training dataset
dummy_features = np.random.rand(7)  # Replace `8` with the number of features in your dataset

# Reshape features to 2D array
dummy_features = dummy_features.reshape(1, -1)

# Predict
try:
    prediction = model.predict(dummy_features)[0]
    confidence = max(model.predict_proba(dummy_features)[0])

    print("Prediction:", "Landslide" if prediction == 1 else "No Landslide")
    print("Confidence:", round(confidence, 2))
except Exception as e:
    print(f"Error during prediction: {e}")
