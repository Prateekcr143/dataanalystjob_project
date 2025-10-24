# src/app_streamlit.py
"""
Simple Streamlit demo to load the trained pipeline and predict salary from user inputs.
Run:
    streamlit run src/app_streamlit.py
"""

import streamlit as st
import joblib
import pandas as pd
import os

MODEL_PATH = "models/rf_salary_model.pkl"

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("Job Posting Salary Predictor")
st.write("Enter job-posting attributes (approximate) and get a predicted average salary.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Train model first by running: python src/train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

# Build inputs dynamically from expected features if available
# We infer expected feature names by inspecting pipeline if possible
try:
    pre = model.named_steps['pre']
    # we created numeric names in train script; fallback to common ones
except Exception:
    pre = None

# Basic inputs (matching train script variables)
rating = st.slider("Company rating (approx.)", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
tech_skill_score = st.number_input("Tech skill count (from job description, e.g., python/sql/tableau...)",
                                   min_value=0, max_value=20, value=2)
jd_length = st.number_input("Job description length (words)", min_value=0, max_value=2000, value=120)
min_salary = st.number_input("Min salary (if known, else estimate)", min_value=0.0, value=0.0)
max_salary = st.number_input("Max salary (if known, else estimate)", min_value=0.0, value=0.0)

# Job title selection - try to populate from pipeline onehot categories if possible
job_title = st.text_input("Job title (type) - use one of the common titles if possible (or 'Other')", value="Data Analyst")
state = st.text_input("State (e.g., CA, NY or 'Unknown')", value="Unknown")

# Prepare DataFrame for model
# This must match the training feature names: 'job_title_clean','state_clean','rating','tech_skill_score','jd_length','min_salary','max_salary'
row = {
    'rating': rating,
    'tech_skill_score': tech_skill_score,
    'jd_length': jd_length,
    'min_salary': min_salary if min_salary>0 else None,
    'max_salary': max_salary if max_salary>0 else None
}

# For category columns we used job_title_clean and state_clean in training script
# Keep naming aligned:
row['job_title_clean'] = job_title
row['state_clean'] = state

df_input = pd.DataFrame([row])

# Replace None with nan so pipeline can handle or preprocess may require numeric; fill min/max if missing with zero
for c in ['min_salary','max_salary']:
    if c in df_input.columns:
        df_input[c] = pd.to_numeric(df_input[c], errors='coerce')

# For rows missing min/max we can impute with median-like fallback (model pipeline expects numeric)
# We do a tiny safe fallback: if both missing, set to 0; if one missing set equal to the other
if pd.isna(df_input.at[0,'min_salary']) and pd.isna(df_input.at[0,'max_salary']):
    df_input['min_salary'] = 0.0
    df_input['max_salary'] = 0.0
elif pd.isna(df_input.at[0,'min_salary']):
    df_input.at[0,'min_salary'] = df_input.at[0,'max_salary']
elif pd.isna(df_input.at[0,'max_salary']):
    df_input.at[0,'max_salary'] = df_input.at[0,'min_salary']

st.write("Input preview:")
st.dataframe(df_input)

if st.button("Predict salary"):
    try:
        pred = model.predict(df_input)[0]
        st.success(f"Predicted average salary: ${pred:,.0f}")
        st.info("This is a model-based estimate. Consider model evaluation metrics printed during training.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
