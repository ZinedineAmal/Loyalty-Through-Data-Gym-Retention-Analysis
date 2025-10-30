import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===========================
# 1. LOAD MODEL DAN SCALER
# ===========================
try:
    model = joblib.load("model_churn.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

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
# 4. KONVERSI INPUT KE DATAFRAME (RAW)
# ===========================
# gunakan nilai string original untuk menghasilkan dummy yang sama seperti training
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
# 5. ENCODING & PREPARE (one-hot seperti training)
# ===========================
# Lakukan one-hot encoding pada kolom kategorikal — ini meniru pd.get_dummies yang kamu pakai di training
categorical_cols = ['gender', 'Near_Location', 'Partner', 'Promo_friends', 'Phone', 'Group_visits']
raw_dummies = pd.get_dummies(raw_input[categorical_cols])

# Ambil kolom numerik (tidak di-dummy)
num_cols = [
    'Contract_period', 'Age', 'Avg_additional_charges_total',
    'Month_to_end_contract', 'Lifetime',
    'Avg_class_frequency_total', 'Avg_class_frequency_current_month'
]
raw_numerical = raw_input[num_cols].astype(float)

# Scale numerik menggunakan scaler yang sudah diload
try:
    scaled_num = pd.DataFrame(scaler.transform(raw_numerical), columns=num_cols, index=raw_input.index)
except Exception as e:
    st.error(f"Terjadi kesalahan saat scaling data: {e}")
    st.stop()

# Gabungkan dummy dan numerik ter-scaling
prepared = pd.concat([raw_dummies, scaled_num], axis=1)

# ===========================
# 6. ALIGN DENGAN FEATURE YANG DIHARAPKAN MODEL
# ===========================
# Dapatkan nama fitur yang dipakai model saat training (atribut ini ada pada estimator sklearn yg sudah fit)
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    # Jika tidak ada, coba muat file feature list jika kamu menyimpannya: 'feature_names.joblib'
    try:
        expected_features = joblib.load("feature_names.joblib")
    except Exception:
        st.error("Model tidak memiliki atribut feature_names_in_ dan file 'feature_names.joblib' tidak ditemukan. "
                 "Simpan daftar nama fitur saat training agar deploy berhasil.")
        st.stop()

# Reindex prepared agar persis sesuai expected_features (kolom yang hilang akan diisi 0)
prepared = prepared.reindex(columns=expected_features, fill_value=0)

# Jika ada kolom tambahan di prepared yang tidak diharapkan model, .reindex di atas sudah membuangnya.

# ===========================
# 7. PREDIKSI
# ===========================
if st.button("🔍 Prediksi Churn"):
    try:
        prediction = model.predict(prepared)[0]
        prob = model.predict_proba(prepared)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Pelanggan **berpotensi CHURN** dengan probabilitas {prob:.2f}")
        else:
            st.success(f"✅ Pelanggan **berpotensi TETAP LOYAL** dengan probabilitas churn {prob:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")


# ===========================
# 8. FOOTER
# ===========================
st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh **Zinedine Amalia** — *Gym Churn Prediction Project*")
