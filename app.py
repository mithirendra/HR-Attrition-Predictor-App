# app.py
# Attrition Predictor — Streamlit Dashboard
# Version 0 — no API

# ------------------------------------
# Section 1 — page config and imports
# ------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

# --- Page config --- must be first Streamlit command
st.set_page_config(
    page_title="Attrition Predictor",
    # page_icon="🎯",
    layout="wide"
)

# --- Version flag ---
API_ENABLED = False

# --- Load precomputed data ---
@st.cache_data
def load_data():
    dept_summary = pd.read_json('data/precomputed/dept_summary.json')
    top10 = pd.read_json('data/precomputed/top10.json')
    dept_trends = pd.read_json('data/precomputed/dept_trends.json')
    emp_trends = pd.read_json('data/precomputed/emp_trends.json')
    with open('data/precomputed/interventions.json') as f:
        interventions = json.load(f)
    feature_importance = pd.read_csv('data/feature_importance.csv')
    return dept_summary, top10, dept_trends, emp_trends, interventions, feature_importance

dept_summary, top10, dept_trends, emp_trends, interventions, feature_importance = load_data()


# ------------------------------------
# Section 2 — Banner, metadata
# ------------------------------------
# --- Banner ---
st.markdown("## Attrition Predictor")
st.markdown("""
An AI-powered tool that identifies which employees are at risk of leaving. 
The model scores each employee by flight risk level, surfaces the key drivers 
behind each score, and recommends targeted HR interventions — giving CHROs and 
people leaders a clear picture of workforce stability before attrition becomes 
a business problem.
""")

col1, col2 = st.columns([3, 1])
with col1:
    st.caption("🟡 Demo mode — synthetic data · 5,000 employees · 5 years · NovaTech Solutions (tech, 5,000 headcount)")
with col2:
    st.success("🟢 Connected to employee master data")

st.divider()

# --- Metadata ---
st.markdown("**Data source:** HRIS · Employee Master")
st.markdown("**Last refreshed:** 16 Apr 2026, 08:00")
st.markdown("**Model:** Logistic Regression · 2021–2025")
st.markdown("**Coverage:** All active employees")
st.markdown("**In production:** connects directly to your HRIS")

st.divider()


# ------------------------------------
# Section 3 — Filters — department, role level, refresh button
# ------------------------------------
# --- Filters ---
col_dept, col_role = st.columns([3, 2])

with col_dept:
    dept_options = ['All'] + sorted(dept_summary['department'].tolist())
    selected_dept = st.selectbox("Department", dept_options)

with col_role:
    role_options = ['All', 'Executive', 'Manager', 'Senior', 'Individual Contributor']
    selected_role = st.selectbox("Role Level", role_options)

# --- Refresh button ---
col1, col2 = st.columns([5, 1])
with col2:
    st.button("🔄 Refresh Data (To enable in future)", help="Connect to live HRIS - to enable this feature in future", disabled=True)



# ------------------------------------
# Section 4 — Metrics — total, high risk, low risk, attrition rate
# ------------------------------------
# --- Metrics ---
total = dept_summary['total'].sum()
high_risk = dept_summary['high_risk'].sum()
low_risk = dept_summary['low_risk'].sum()
attrition_rate = (high_risk / total * 100).round(1)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Employees", f"{total:,}")
m2.metric("High Risk", f"{high_risk:,}")
m3.metric("Low Risk", f"{low_risk:,}")
m4.metric("Predicted Attrition Rate", f"{attrition_rate}%")

st.divider()


# ------------------------------------
# Section 5 — Employee flight risk table
# ------------------------------------
# --- Section 5: Employee Flight Risk Table ---
st.markdown("### Employee Flight Risk — MM Group")
st.caption("Click column headers to sort. Filter by department and role level above.")

# Apply filters to top10 data
filtered = top10.copy()

if selected_dept != 'All':
    filtered = filtered[filtered['department'] == selected_dept]

if selected_role != 'All':
    filtered = filtered[filtered['role_level'] == selected_role]

filtered['tenure'] = filtered['tenure'].round(1)

# Select display columns
display_cols = ['employee_id', 'department', 'role_level', 'tenure', 
                'risk_level', 'risk_score']

# Style risk level column
def color_risk(val):
    if val == 'High':
        return 'background-color: #FCEBEB; color: #A32D2D; font-weight: bold'
    return 'background-color: #EAF3DE; color: #3B6D11; font-weight: bold'

styled = filtered[display_cols].style.map(
    color_risk, subset=['risk_level']
).set_properties(subset=['tenure', 'risk_level', 'risk_score'], **{'text-align': 'center'})


st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "tenure": st.column_config.NumberColumn("Tenure", format="%.1f"),
        "risk_score": st.column_config.NumberColumn("Risk Score"),
        "risk_level": st.column_config.TextColumn("Risk Level"),
    }
)

st.divider()

# ------------------------------------
# Section 6 — 5-year behavioural trend
# ------------------------------------
# --- Section 6: 5-Year Behavioural Trend ---
st.markdown("### 5-Year Behavioural Trend")

trend_cols = ['engagement_score', 'engagement_activity', 'online_learning',
              'f2f_learning', 'absenteeism', 'overtime_hours']
col1, col2 = st.columns([3, 2])

with col1:
    if selected_dept != 'All':
        trend_dept = selected_dept
    else:
        trend_dept = dept_trends['department'].iloc[0]
    st.caption(f"Department: {trend_dept}")

with col2:
    # Employee selector
    emp_options = ['All (department average)'] + filtered['employee_id'].tolist()
    selected_emp = st.selectbox("Select employee", emp_options)

# Show individual or department average
if selected_emp == 'All (department average)':
    trend_data = dept_trends[dept_trends['department'] == trend_dept].sort_values('year')
    st.caption(f"Showing high risk employees — {trend_dept} · avg per year")
else:
    # Load full dataset for individual employee
    trend_data = emp_trends[emp_trends['employee_id'] == selected_emp][['year'] + trend_cols].sort_values('year')
    st.caption(f"Showing individual trend — {selected_emp}")

trend_display = trend_data.set_index('year').drop(columns=['department'], errors='ignore')
trend_display.columns = ['Engagement Score', 'Engagement Activity',
                          'Online Learning', 'F2F Learning',
                          'Absenteeism', 'Overtime Hours']

st.dataframe(trend_display, use_container_width=True)
st.divider()


# ------------------------------------
# Section 7 — Top attrition driver
# ------------------------------------
# --- Section 7: Top Attrition Drivers ---
st.markdown("### Top Attrition Drivers")

# Show top 10 drivers
top_drivers = feature_importance.head(10)

for _, row in top_drivers.iterrows():
    col1, col2, col3 = st.columns([3, 6, 1])
    with col1:
        st.markdown(f"**{row['feature'].replace('_', ' ').title()}**")
    with col2:
        st.progress(float(row['importance']) / float(feature_importance['importance'].max()))
    with col3:
        st.caption(f"{row['importance']:.3f}")

st.divider()


# ------------------------------------
# Section 8 — Department heatmap
# ------------------------------------
# --- Section 8: Department Heatmap ---
st.markdown("### Department Heatmap")


# Pivot for display
heatmap_data = dept_summary[['department', 'high_risk_pct', 'low_risk_pct']].copy()
heatmap_data.columns = ['Department', 'High Risk %', 'Low Risk %']
heatmap_data = heatmap_data.sort_values('High Risk %', ascending=False)

heatmap_data['High Risk %'] = heatmap_data['High Risk %'].round(1)
heatmap_data['Low Risk %'] = heatmap_data['Low Risk %'].round(1)

if selected_dept != 'All':
    heatmap_data = heatmap_data[heatmap_data['Department'] == selected_dept]

# Color high risk column
def color_high_risk(val):
    if val >= 25:
        return 'background-color: #FCEBEB; color: #A32D2D; font-weight: bold'
    elif val >= 15:
        return 'background-color: #FAEEDA; color: #854F0B; font-weight: bold'
    return 'background-color: #EAF3DE; color: #3B6D11; font-weight: bold'

styled_heatmap = heatmap_data.style.map(
    color_high_risk, subset=['High Risk %']
)

st.dataframe(
    styled_heatmap,
    use_container_width=True,
    hide_index=True,
    column_config={
        "High Risk %": st.column_config.NumberColumn("High Risk %", format="%.1f"),
        "Low Risk %": st.column_config.NumberColumn("Low Risk %", format="%.1f")
    }
)

st.divider()


# ------------------------------------
# Section 9 — Interventions:
# ------------------------------------
# --- Section 9: Interventions ---
st.markdown("### Recommended Interventions by Department")
st.caption(f"Generated by rule-based model · Version 0")

# Filter by selected department
if selected_dept != 'All':
    depts_to_show = [selected_dept]
else:
    depts_to_show = list(interventions.keys())

for dept in depts_to_show:
    dept_row = dept_summary[dept_summary['department'] == dept].iloc[0]
    high_pct = dept_row['high_risk_pct']
    
    if high_pct >= 25:
        st.error(f"**{dept}** · {high_pct}% high risk · immediate action required")
    elif high_pct >= 15:
        st.warning(f"**{dept}** · {high_pct}% high risk · action required")
    elif high_pct >= 10:
        st.warning(f"**{dept}** · {high_pct}% high risk · monitor closely")
    else:
        st.success(f"**{dept}** · {high_pct}% high risk · stable")
    
    st.markdown(interventions[dept])
    st.divider()

    # --- Footer ---
st.markdown("<p style='text-align:center; color:grey;'>© 2026 Version 0 Built by Mithirendra Maniam</p>", unsafe_allow_html=True)