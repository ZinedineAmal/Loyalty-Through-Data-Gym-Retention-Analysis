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
    'Avg_class_frequency_current_month': [avg_class_frequency_current_month],
})

# ===========================
# 5. ENCODING & PREPARATION (one-hot encoding as in training)
# ===========================
categorical_cols = ['gender', 'Near_Location', 'Partner', 'Promo_friends', 'Phone', 'Group_visits']
raw_dummies = pd.get_dummies(raw_input[categorical_cols])

num_cols = [
    'Contract_period', 'Age', 'Avg_additional_charges_total',
    'Month_to_end_contract', 'Lifetime',
    'Avg_class_frequency_total', 'Avg_class_frequency_current_month'
]
raw_numerical = raw_input[num_cols].astype(float)

# Scale numeric features
try:
    scaled_num = pd.DataFrame(scaler.transform(raw_numerical), columns=num_cols, index=raw_input.index)
except Exception as e:
    st.error(f"Error while scaling data: {e}")
    st.stop()

# Combine dummies and scaled numeric features
prepared = pd.concat([raw_dummies, scaled_num], axis=1)

# ===========================
# 6. ALIGN FEATURES WITH MODEL
# ===========================
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    try:
        expected_features = joblib.load("feature_names.joblib")
    except Exception:
        st.error(
            "Model does not contain 'feature_names_in_' and no 'feature_names.joblib' file found. "
            "Please save the feature list during training for consistent deployment."
        )
        st.stop()

# Align column names (missing ones will be filled with 0)
prepared = prepared.reindex(columns=expected_features, fill_value=0)

# ===========================
# 7. PREDICTION
# ===========================
if st.button("🔍 Predict Churn"):
    try:
        prediction = model.predict(prepared)[0]
        prob = model.predict_proba(prepared)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The customer is **likely to CHURN**, with a probability of {prob:.2f}")
        else:
            st.success(f"✅ The customer is **likely to REMAIN LOYAL**, with a churn probability of {prob:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ===========================
# 8. FOOTER
# ===========================
st.markdown("---")
st.markdown("Built with ❤️ by **Zinedine Amalia** — *Gym Churn Prediction Project*")
