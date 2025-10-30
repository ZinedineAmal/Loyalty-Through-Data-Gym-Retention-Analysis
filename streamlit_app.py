import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===========================
# 1. LOAD MODEL AND SCALER
# ===========================
try:
    model = joblib.load("model_churn.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    st.stop()

# ===========================
# 2. APP HEADER
# ===========================
st.set_page_config(page_title="Gym Churn Prediction", layout="centered")
st.title("🏋️‍♀️ Gym Customer Churn Prediction App")
st.markdown(
    "Fill in the customer details below to predict whether a member is **likely to churn (leave)** or **remain loyal**."
)

# ===========================
# 3. INPUT FORM
# ===========================
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    near_location = st.selectbox("Lives near the gym?", ["Yes", "No"])
    partner = st.selectbox("Has a workout partner?", ["Yes", "No"])
    promo_friends = st.selectbox("Joined through friends' promotion?", ["Yes", "No"])
    phone = st.selectbox("Provided phone number?", ["Yes", "No"])
    group_visits = st.selectbox("Attends group classes?", ["Yes", "No"])

with col2:
    contract_period = st.number_input("Contract Period (months)", min_value=1, max_value=24, value=6)
    age = st.number_input("Age", min_value=16, max_value=70, value=30)
    avg_additional_charges_total = st.number_input("Average Additional Charges (USD)", min_value=0.0, value=50.0)
    lifetime = st.number_input("Membership Duration (months)", min_value=1, max_value=60, value=12)
    avg_class_frequency_total = st.number_input("Average Total Visit Frequency", min_value=0.0, max_value=10.0, value=2.0)
    month_to_end_contract = st.number_input("Months Remaining in Contract", min_value=0, max_value=24, value=3)
    avg_class_frequency_current_month = st.number_input("Average Frequency This Month", min_value=0.0, max_value=10.0, value=1.5)

# ===========================
# 4. CONVERT INPUT TO DATAFRAME (RAW)
# ===========================
raw_input = pd.DataFrame({
    'gender': [gender],
    'Near_Location': [near_location],
    'Partner': [partner],
    'Promo_friends': [promo_friends],
    'Phone': [phone],
    'Group_visits': [group_visits],
    'Contract_period': [contract_period],
    'Age': [age],
    'Avg_additional_charges_total': [avg_additional_charges_total],
    'Month_to_end_contract': [month_to_end_contract],
    'Lifetime': [lifetime],
    'Avg_class_frequency_total': [avg_class_frequency_total],
