import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Step 1: Load the training and test datasets
train_data_path = '/Users/avinbennyk/Desktop/Landslidepro/Dataset/train_data.csv'
test_data_path = '/Users/avinbennyk/Desktop/Landslidepro/Dataset/test_data.csv'

# Load datasets
try:
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Step 2: Split into features (X) and target (y)
X_train = train_data.drop(columns=['Landslide'])
y_train = train_data['Landslide']

X_test = test_data.drop(columns=['Landslide'])
y_test = test_data['Landslide']

# Step 3: Apply SMOTE to rebalance the training dataset
print("\nApplying SMOTE to rebalance the dataset...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 4: Hyperparameter Tuning for Random Forest
print("\nTuning Random Forest hyperparameters on SMOTE-balanced data...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

rf_grid_search.fit(X_train_balanced, y_train_balanced)
best_rf_model = rf_grid_search.best_estimator_
print("\nBest Random Forest Hyperparameters:")
print(rf_grid_search.best_params_)

# Step 5: Hyperparameter Tuning for XGBoost
print("\nTuning XGBoost hyperparameters on SMOTE-balanced data...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [4, 5],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [1, 3]
}

xgb_grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
    param_grid=xgb_param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

xgb_grid_search.fit(X_train_balanced, y_train_balanced)
best_xgb_model = xgb_grid_search.best_estimator_
print("\nBest XGBoost Hyperparameters:")
print(xgb_grid_search.best_params_)

# Step 6: Evaluate Individual Models
print("\nCross-Validation for Individual Models:")
cv_scores_rf = cross_val_score(best_rf_model, X_train_balanced, y_train_balanced, cv=3, scoring='accuracy')
print(f"Random Forest CV Accuracy: {np.mean(cv_scores_rf):.2f}")

cv_scores_xgb = cross_val_score(best_xgb_model, X_train_balanced, y_train_balanced, cv=3, scoring='accuracy')
print(f"XGBoost CV Accuracy: {np.mean(cv_scores_xgb):.2f}")

# Step 7: Stacking Model
print("\nCreating Stacking Model...")
stacking_model = StackingClassifier(
    estimators=[
        ('random_forest', best_rf_model),
        ('xgboost', best_xgb_model)
    ],
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train_balanced, y_train_balanced)

# Evaluate Stacking Model
print("\nEvaluating Stacking Model on Test Data...")
y_pred_stacking = stacking_model.predict(X_test)
print(classification_report(y_test, y_pred_stacking))

# Step 8: Save the Best Model (Stacking)
model_directory = "/Users/avinbennyk/Desktop/Landslidepro/Model"
os.makedirs(model_directory, exist_ok=True)

model_filename = os.path.join(model_directory, "best_stacking_model.pkl")
joblib.dump(stacking_model, model_filename)
print(f"Best Stacking Model saved to {model_filename}")

# Step 9: Feature Importance Analysis
print("\nFeature Importance Analysis:")
feature_names = X_train.columns
importances_rf = best_rf_model.feature_importances_
importances_xgb = best_xgb_model.feature_importances_

average_importances = (importances_rf + importances_xgb) / 2

plt.figure(figsize=(10, 6))
plt.barh(feature_names, average_importances, align='center')
plt.xlabel("Average Feature Importance")
plt.ylabel("Feature")
plt.title("Average Feature Importance Analysis")
plt.show()
