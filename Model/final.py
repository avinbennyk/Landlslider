import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv( '/Users/avinbennyk/Desktop/Landslidepro/Dataset/preprocessed_data.csv')
X = data.drop('Landslide', axis=1)
y = data['Landslide']

# Feature engineering with polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Balancing the training data with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Hyperparameter grids
params_rf = {'n_estimators': [100, 200], 'max_depth': [10, 15, 20]}
params_xgb = {'n_estimators': [50, 100], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]}

# Grid search for Random Forest and XGBoost
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params_rf, cv=3, scoring='accuracy')
grid_xgb = GridSearchCV(XGBClassifier(subsample=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42),
                        param_grid=params_xgb, cv=3, scoring='accuracy')

# Fit grid search
grid_rf.fit(X_train_balanced, y_train_balanced)
grid_xgb.fit(X_train_balanced, y_train_balanced)

# Best estimators
best_rf = grid_rf.best_estimator_
best_xgb = grid_xgb.best_estimator_

# Define base models with the best parameters from grid search
base_models = [
    ('rf', best_rf),
    ('svm', SVC(kernel='rbf', C=1, gamma=0.1, probability=True, random_state=42)),
    ('xgb', best_xgb)
]

# Stacking ensemble with logistic regression as a meta-learner
stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), passthrough=True, cv=5)
stack.fit(X_train_balanced, y_train_balanced)

# Evaluating the model
predictions = stack.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))


# Save the trained model
model_save_path = '/Users/avinbennyk/Desktop/Landslidepro/Model/best_stacking_model.pkl'
joblib.dump(stack, model_save_path)
print(f"Model saved to {model_save_path}")