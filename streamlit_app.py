import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================
# FILE PATH SESUAI NAMA FILE KAMU
# ===========================================
MODEL_PATH = "model_churn.pkl"
PREPROCESS_PATH = "preprocess.pkl"
DATA_PATH = "clean_df.csv"

# ===========================================
# VALIDASI FILE
# ===========================================
for f in [MODEL_PATH, PREPROCESS_PATH, DATA_PATH]:
    if not os.path.exists(f):
        st.error(f"File tidak ditemukan: {f}")
        st.stop()

# ===========================================
# LOAD ASSET
# ===========================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocess():
    return joblib.load(PREPROCESS_PATH)

@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
preprocess = load_preprocess()
df = load_data()

# ===========================================
# IDENTIFIKASI INPUT FEATURES SEBELUM PREPROCESS
# ===========================================
# (ambil semua kolom kecuali churn)
target_col = "customer_status"
input_cols = [c for c in df.columns if c != target_col]

# pisahkan tipe numerik & kategori
num_cols = df[input_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df[input_cols].select_dtypes(include=["object", "category"]).columns.tolist()

# ===========================================
# SIDEBAR MENU
# ===========================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Prediction", "EDA"])


# ===========================================
# PAGE 1 — PREDICTION
# ===========================================
if page == "Prediction":

    st.title("Customer Churn Prediction")
    st.write("Isi informasi customer untuk prediksi churn.")

    input_data = {}

    with st.form("form_customer"):

        st.subheader("Customer Details")

        # Input numerik
        for col in num_cols:
            default_val = float(df[col].median())
            input_data[col] = st.number_input(col, value=default_val)

        # Input kategorikal
        for col in cat_cols:
            options = sorted(df[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("Predict")

    if submitted:

        input_df = pd.DataFrame([input_data])

        # PREPROCESS (langsung transform)
        processed = preprocess.transform(input_df)
        processed_df = pd.DataFrame(
            processed,
            columns=preprocess.get_feature_names_out()
        )

        # PREDICT
        pred = model.predict(processed_df)[0]
        prob = model.predict_proba(processed_df)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"Customer BERISIKO churn. Probability: {prob:.2f}")
        else:
            st.success(f"Customer TIDAK churn. Probability: {prob:.2f}")


# ===========================================
# PAGE 2 — EDA
# ===========================================
else:
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")

    st.markdown("---")

    # Distribusi numerik
    st.subheader("Numerical Feature Distribution")
    selected_num = st.selectbox("Pilih kolom numerik:", num_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # Distribusi kategorik
    st.subheader("Categorical Feature Distribution")
    selected_cat = st.selectbox("Pilih kolom kategorik:", cat_cols)

    fig2, ax2 = plt.subplots()
    df[selected_cat].value_counts().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df[num_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.success("EDA selesai.")
