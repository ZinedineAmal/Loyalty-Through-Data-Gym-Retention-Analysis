import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===========================
# 1. LOAD MODEL DAN SCALER
# ===========================
with open("model_churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===========================
# 2. HEADER APLIKASI
# ===========================
st.set_page_config(page_title="Gym Churn Prediction", layout="centered")
st.title("🏋️‍♀️ Gym Customer Churn Prediction App")
st.markdown("Masukkan detail pelanggan untuk memprediksi apakah pelanggan akan **churn (berhenti)** atau **tetap loyal**.")

# ===========================
# 3. INPUT FORM
# ===========================
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    near_location = st.selectbox("Tinggal dekat lokasi gym?", ["Yes", "No"])
    partner = st.selectbox("Punya partner latihan?", ["Yes", "No"])
    promo_friends = st.selectbox("Dapat promo dari teman?", ["Yes", "No"])
    phone = st.selectbox("Memberikan nomor telepon?", ["Yes", "No"])
    group_visits = st.selectbox("Ikut kelas grup?", ["Yes", "No"])

with col2:
    contract_period = st.number_input("Lama Kontrak (bulan)", min_value=1, max_value=24, value=6)
    age = st.number_input("Umur", min_value=16, max_value=70, value=30)
    avg_additional_charges_total = st.number_input("Rata-rata Biaya Tambahan (USD)", min_value=0.0, value=50.0)
    lifetime = st.number_input("Lama Menjadi Anggota (bulan)", min_value=1, max_value=60, value=12)
    avg_class_frequency_total = st.number_input("Rata-rata Kehadiran Total", min_value=0.0, max_value=10.0, value=2.0)
    month_to_end_contract = st.number_input("Sisa Bulan Kontrak", min_value=0, max_value=24, value=3)
    avg_class_frequency_current_month = st.number_input("Rata-rata Kehadiran Bulan Ini", min_value=0.0, max_value=10.0, value=1.5)

# ===========================
# 4. KONVERSI INPUT KE DATAFRAME
# ===========================
input_data = pd.DataFrame({
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
# 5. ENCODING MANUAL SESUAI TRAINING
# ===========================
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
for col in ['Near_Location', 'Partner', 'Promo_friends', 'Phone', 'Group_visits', 'gender']:
    input_data[col] = input_data[col].map(binary_map)

# ===========================
# 6. SCALE FITUR NUMERIK
# ===========================
num_cols = ['Contract_period', 'Age', 'Avg_additional_charges_total',
            'Month_to_end_contract', 'Lifetime',
            'Avg_class_frequency_total', 'Avg_class_frequency_current_month']

input_data[num_cols] = scaler.transform(input_data[num_cols])

# ===========================
# 7. PREDIKSI
# ===========================
if st.button("🔍 Prediksi Churn"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Pelanggan **berpotensi CHURN** dengan probabilitas {prob:.2f}")
    else:
        st.success(f"✅ Pelanggan **berpotensi TETAP LOYAL** dengan probabilitas churn {prob:.2f}")

# ===========================
# 8. FOOTER
# ===========================
st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh Zinedine Amalia — Gym Churn Prediction Project")

