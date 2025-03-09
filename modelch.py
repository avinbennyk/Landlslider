import joblib

# Specify the path from where to load the model
model_load_path = '/Users/avinbennyk/Desktop/Landslidepro/Model/best_stacking_model.pkl'
loaded_model = joblib.load(model_load_path)

# Print the type of the loaded model
print("Type of loaded model:", type(loaded_model))
