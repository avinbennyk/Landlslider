import joblib
model_path = "/Users/avinbennyk/Desktop/Landslidepro/Model/best_stacking_model.pkl"

with open(model_path, "rb") as f:
    stacking_model = joblib.load(f)

# Test the model on sample data
sample_input = [[...]]  # Replace with valid feature values
print("Prediction:", stacking_model.predict(sample_input))
