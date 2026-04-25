# train_model.py
# Attrition Predictor — Model Training
# Uses Logistic Regression — industry standard for HR attrition prediction
# Produces natural probability scores interpretable by CHRO audience

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# Step 1 — Load data
# --------------------------------------------------
df = pd.read_csv('data/employee_data.csv')


# --------------------------------------------------
# Step 2 — Encode categorical columns
# --------------------------------------------------
# Encode categorical columns to numbers
le_gender = LabelEncoder()
le_dept = LabelEncoder()
le_role = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['department'] = le_dept.fit_transform(df['department'])
df['role_level'] = le_role.fit_transform(df['role_level'])

# --------------------------------------------------
# Step 3 — Prepare features and target
# --------------------------------------------------
# Get X and y values
X = df.drop(['employee_id','attrition', 'year'], axis=1)
y = df['attrition']

# Scale values to be closer to each other
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

print("Step 1 - 3 Done")

# Checking number of observations
N_observations, N_features = X_std.shape
print('Number of Observations: ' + str(N_observations))
print('Number of Features: ' + str(N_features))

# --------------------------------------------------
# Step 4 — Train Logistic Regression
# --------------------------------------------------
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)


# --------------------------------------------------
# Step 5 — Evaluate model
# --------------------------------------------------
y_pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Check model accuracy
# Calculate the MSE and R^2 values for both models - this is wrong, this is used for regressor models, but model below is Classifier.

# RandomForestRegressor vs. RandomForestClassifier.
# The difference:

# Regressor — predicts a continuous number like salary, temperature, house price
# Classifier — predicts a category like yes/no, high/medium/low

# Attrition is yes/no — that's a classification problem, not regression.
# Also the accuracy metrics need to change:


# --------------------------------------------------
# Step 5b — Feature importance from LR coefficients
# --------------------------------------------------
print("\nTop Attrition Drivers (Feature Importance):")
feature_importance_df = pd.DataFrame({
    'feature': X.columns.tolist(),
    'importance': abs(lr.coef_[0])
}).sort_values('importance', ascending=False)

print(feature_importance_df)


# --------------------------------------------------
# Step 6 — SHAP values
# --------------------------------------------------
import shap

print("\nCalculating SHAP values - this will take 1-2 minutes...")

# Create SHAP explainer using the trained model
explainer = shap.LinearExplainer(lr, X_train)

# Calculate SHAP values for the test set
# We use test set only - faster and sufficient for our purposes
shap_values = explainer.shap_values(X_test)

# Calculate mean absolute SHAP value per feature - global importance
mean_shap = pd.DataFrame({
    'feature': X.columns.tolist(),
    'importance': abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTop Attrition Drivers (SHAP):")
print(mean_shap)

# Feature Importance counts how many times a feature is used across all 100 decision trees. Engagement score gets used most frequently.
# SHAP measures actual impact on the final prediction. Manager score has a smaller but more decisive impact when it appears.
# Which to trust more:
# SHAP. It measures actual impact on predictions, not just usage frequency. A feature can be used often but weakly — Feature Importance would rank it high, SHAP would rank it lower.


# Step 7 - Save the model - saving the model so Streamlit can load it without retraining every time.
import joblib
import numpy as np

print("\nSaving model and supporting files...")

# Save the trained Logistic Regression model
joblib.dump(lr, 'models/lr_model.pkl')

# Save the scaler — needed to scale new data the same way
joblib.dump(std_scaler, 'models/scaler.pkl')

# Save the label encoders — needed to encode new data the same way
joblib.dump(le_gender, 'models/le_gender.pkl')
joblib.dump(le_dept, 'models/le_dept.pkl')
joblib.dump(le_role, 'models/le_role.pkl')

# Save SHAP values and feature names for the dashboard
np.save('models/shap_values.npy', shap_values)
mean_shap.to_csv('data/feature_importance.csv', index=False)

# Save feature names
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("All files saved to models/ folder")