# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Update the file path to match your local environment
train_data_path = '/Users/avinbennyk/Desktop/Landslidepro/Dataset/train_data.csv'

# Load the dataset
try:
    train_data = pd.read_csv(train_data_path)
except FileNotFoundError:
    print(f"File not found at {train_data_path}. Please check the path and try again.")
    exit()

# Split the dataset into features (X) and target (y)
X = train_data.drop(columns=['Landslide'])
y = train_data['Landslide']

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 1: Initial Random Forest Model
print("Training initial Random Forest model...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_val)
print("\nInitial Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.2f}")
print(classification_report(y_val, y_pred))

# Step 2: Hyperparameter Tuning with GridSearchCV
print("\nTuning hyperparameters...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predict and evaluate the best model
y_pred_best = best_model.predict(X_val)
print("\nBest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_best):.2f}")
print(classification_report(y_val, y_pred_best))

# Step 3: Feature Importance Analysis
print("\nFeature Importances:")
feature_importances = best_model.feature_importances_
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, align='center')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Analysis")
plt.show()
