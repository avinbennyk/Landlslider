# Import necessary libraries
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Step 1: Load the training and test datasets
train_data_path = '/Users/avinbennyk/Desktop/Landslidepro/Dataset/train_data.csv'  # Update the path
test_data_path = '/Users/avinbennyk/Desktop/Landslidepro/Dataset/test_data.csv'    # Update the path

# Load datasets
try:
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
except FileNotFoundError as e:
    print(e)
    exit()

# Step 2: Split into features (X) and target (y)
X_train = train_data.drop(columns=['Landslide'])
y_train = train_data['Landslide']

X_test = test_data.drop(columns=['Landslide'])
y_test = test_data['Landslide']

# Step 3: Define the parameter grid for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],     # Number of trees
    'learning_rate': [0.01, 0.1, 0.2], # Step size shrinkage
    'max_depth': [3, 5, 7],            # Maximum depth of a tree
    'subsample': [0.8, 1.0],           # Subsample ratio of the training instances
    'colsample_bytree': [0.8, 1.0]     # Subsample ratio of columns when constructing each tree
}

# Initialize the XGBoost classifier
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")

# Step 4: Perform GridSearchCV for hyperparameter tuning
print("Running hyperparameter tuning for XGBoost...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation
    scoring='accuracy', # Optimize for accuracy
    verbose=1,          # Show progress
    n_jobs=-1           # Use all CPU cores
)
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_xgb_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Step 5: Evaluate the model on the test data
y_pred_test = best_xgb_model.predict(X_test)

print("\nBest Hyperparameters:")
print(best_params)

print("\nXGBoost Model Evaluation on Test Data:")
xgb_accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {xgb_accuracy_test:.2f}")
print(classification_report(y_test, y_pred_test))

# Step 6: Visualize Feature Importances
print("\nFeature Importances (Tuned XGBoost):")
xgb_importances_tuned = best_xgb_model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, xgb_importances_tuned, align='center')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Analysis - Tuned XGBoost")
plt.show()
