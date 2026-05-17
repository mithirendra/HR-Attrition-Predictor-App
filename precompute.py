
# ------------------------------------------------
# SECTION 1: Imports and setup
# ------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import json
import os

# Load all saved model files
print('Loading model files.....')
lr = joblib.load('models/lr_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le_gender = joblib.load('models/le_gender.pkl')
le_dept = joblib.load('models/le_dept.pkl')
le_role = joblib.load('models/le_role.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Load employee data
df = pd.read_csv('data/employee_data.csv')

# Load feature importance
feature_importance = pd.read_csv('data/feature_importance.csv')

# Output folder for precomputed data
os.makedirs('data/precomputed', exist_ok=True)

print("All files loaded successfully")


# ------------------------------------------------
# Section 2: Encode data and run model predictions
# ------------------------------------------------
print("Running model predictions...")

# Make a copy so we don't change the original data
df_encoded = df.copy()
df_encoded['employee_id'] = df['employee_id']

# Encode categorical columns using the same encoders from training
df_encoded['gender'] = le_gender.transform(df_encoded['gender'])
df_encoded['department'] = le_dept.transform(df_encoded['department'])
df_encoded['role_level'] = le_role.transform(df_encoded['role_level'])

# Get the latest year data only (2025) for risk scoring
# We score employees based on their most recent data
df_2025 = df_encoded[df_encoded['year'] == 2025].copy()

# Prepare features — same columns as training
X = df_2025[feature_names]

# Scale features using the same scaler from training
X_scaled = scaler.transform(X)

# Get prediction probabilities — probability of leaving (class 1)
probabilities = lr.predict_proba(X_scaled)[:, 1]

# Add risk score (0-100) to the 2025 dataframe
df_2025['risk_score'] = (probabilities * 100).round(1)

# Assign risk levels based on score thresholds
df_2025['risk_level'] = pd.cut(
    df_2025['risk_score'],
    bins=[-0.1, 50, 100],
    labels=['Low', 'High']
)

print(f"Predictions complete — {len(df_2025)} employees scored")
print(f"High risk: {(df_2025['risk_level'] == 'High').sum()}")
print(f"Low risk: {(df_2025['risk_level'] == 'Low').sum()}")


# ------------------------------------------------
# Section 3: Compute department summaries
# ------------------------------------------------
print("Computing department summaries...")

dept_summary = df_2025.groupby('department').agg(
    total=('employee_id', 'count'),
    high_risk=('risk_level', lambda x: (x == 'High').sum()),
).reset_index()

dept_summary['high_risk_pct'] = (dept_summary['high_risk'] / dept_summary['total'] * 100).round(1)
dept_summary['low_risk'] = dept_summary['total'] - dept_summary['high_risk']
dept_summary['low_risk_pct'] = (dept_summary['low_risk'] / dept_summary['total'] * 100).round(1)

# Map department numbers back to names
dept_names = dict(zip(le_dept.transform(le_dept.classes_), le_dept.classes_))
dept_summary['department'] = dept_summary['department'].map(dept_names)
print(dept_summary)

# ------------------------------------------------
# Section 4: Top 10 high risk employees per department
# ------------------------------------------------
print("Computing top 10 high risk employees per department...")

# Filter high risk only, sort by highest score, take top 10 per department
top10 = df_2025[df_2025['risk_level'] == 'High'].sort_values(
    'risk_score', ascending=False
).groupby('department').head(10).reset_index(drop=True)

# Map encoded columns back to readable names
top10['department'] = top10['department'].map(dept_names)
top10['gender'] = le_gender.inverse_transform(top10['gender'].astype(int))
top10['role_level'] = le_role.inverse_transform(top10['role_level'].astype(int))

top10['tenure'] = top10['tenure'].round(1)

print(f"Top 10 per department computed — {len(top10)} total rows")


# ------------------------------------------------
# Section 5: 5-year trend data per department
# ------------------------------------------------
print("Computing 5-year trend data...")

# Get high risk employee IDs from 2025 scoring
high_risk_ids = df_2025[df_2025['risk_level'] == 'High']['employee_id'].tolist()

# Filter full 5-year dataset to high risk employees only — get their complete history
df_high_risk = df_encoded[df_encoded['employee_id'].isin(high_risk_ids)].copy()

# Columns to track over time
trend_cols = ['engagement_score', 'engagement_activity', 'online_learning',
              'f2f_learning', 'absenteeism', 'overtime_hours']

# Average each metric per department per year
dept_trends = df_high_risk.groupby(['department', 'year'])[trend_cols].mean().round(2).reset_index()

# Map dept trends
dept_trends['department'] = dept_trends['department'].map(dept_names)
print(dept_trends)

print(f"Trend data computed — {len(dept_trends)} rows")


# ------------------------------------------------
# Section 6: Rule-based interventions
# -----------------------------------------------
print("Generating rule-based interventions.......")

# Intervention logic based on department risk profile
def generate_intervention(dept, high_risk_pct, top_drivers):
    
    if high_risk_pct >= 25:
        urgency = "immediate action required"
    elif high_risk_pct >= 15:
        urgency = "action required"
    elif high_risk_pct >= 10:
        urgency = "monitor closely"
    else:
        urgency = "stable"
    
    if urgency == "stable":
        return f"{dept} is performing well below company average attrition risk. Engagement scores are strong. Continue current people practices — share retention strategies with higher-risk departments."
    
    interventions = []
    
    if 'manager_score' in top_drivers:
        interventions.append("Conduct manager effectiveness reviews and provide targeted coaching.")
    if 'engagement_score' in top_drivers:
        interventions.append("Launch engagement pulse survey and conduct stay interviews.")
    if 'months_since_promotion' in top_drivers:
        interventions.append("Review promotion pipeline — flag employees with 24+ months without progression.")
    if 'absenteeism' in top_drivers:
        interventions.append("Investigate absenteeism patterns — refer high-absence employees to HR business partners.")
    if 'overtime_hours' in top_drivers:
        interventions.append("Audit workload distribution — sustained overtime is a burnout signal.")
    if 'performance_rating' in top_drivers:
        interventions.append("Review performance management process — check for rating inflation or bias.")
    
    return f"{dept} has {high_risk_pct}% high risk employees — {urgency}. " + " ".join(interventions)

# Get top 3 drivers from feature importance
top_drivers = feature_importance['feature'].head(3).tolist()

# Generate intervention per department
interventions = {}
for _, row in dept_summary.iterrows():
    interventions[row['department']] = generate_intervention(
        row['department'],
        row['high_risk_pct'],
        top_drivers
    )

print("Interventions generated")
for dept, text in interventions.items():
    print(f"\n{dept}: {text}")


# -----------------------------------------------
# Section 7: Save everything to JSON
# -----------------------------------------------
print("\nSaving precomputed data...")

# Save department summary
dept_summary.to_json('data/precomputed/dept_summary.json', orient='records', indent=2)

# Save top 10 high risk per department
top10.to_json('data/precomputed/top10.json', orient='records', indent=2)

# Save trend data
dept_trends.to_json('data/precomputed/dept_trends.json', orient='records', indent=2)

# Save individual employee 5-year trends
emp_trends = df[df['employee_id'].isin(high_risk_ids)][
    ['employee_id', 'year'] + trend_cols
].copy()
emp_trends.to_json('data/precomputed/emp_trends.json', orient='records', indent=2)

# Save interventions
with open('data/precomputed/interventions.json', 'w') as f:
    json.dump(interventions, f, indent=2)

print("All precomputed data saved to data/precomputed/")