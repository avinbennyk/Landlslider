import joblib
import numpy as np

# Load the stacking model
rf_model = joblib.load("/Users/avinbennyk/Desktop/Landslidepro/Model/best_stacking_model.pkl")

def predict(features):
    """
    Predict landslide risk using the loaded stacking model.
    """
    try:
        # Reshape features to 2D array
        features = np.array(features).reshape(1, -1)

        # Predict
        prediction = rf_model.predict(features)[0]
        confidence = max(rf_model.predict_proba(features)[0])

        return {
            "prediction": "Landslide" if prediction == 1 else "No Landslide",
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
