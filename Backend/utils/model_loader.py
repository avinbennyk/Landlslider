import pickle
import numpy as np

# Load the models
rf_model = pickle.load(open("../Model/best_stacking_model.pkl", "rb"))

def predict(features):
    """
    Predict landslide risk using the preloaded models.
    """
    features = np.array(features).reshape(1, -1)
    prediction = rf_model.predict(features)[0]
    confidence = max(rf_model.predict_proba(features)[0])

    return {
        "model": "Random Forest",
        "prediction": "Landslide" if prediction == 1 else "No Landslide",
        "confidence": round(confidence, 1)
    }
