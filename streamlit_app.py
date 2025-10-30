import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============== LOAD MODEL & SCALER ==============
try:
    model = joblib.load("model_churn.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    st.stop()

# ============== APP HEADER ==============
st.set_page_config(page_title="Gym Churn Dashboard", layout="wide")
st.title("🏋️‍♀️ Gym Customer Churn Dashboard")
st.markdown("Explore customer insights and predict churn probabilities.")

# ============== LOAD DATASET ==============
@st.cache_data
def load_data():
    df = pd.read_csv("gym_churn_us.csv")
    loyal = df[df["Churn"] == 0].copy()
    return df, loyal

df, loyal = load_data()

# ============== TAB STRUCTURE ==============
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🔮 Churn Prediction"])

# ============== TAB 1: EDA ==============
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Below are the key insights from the loyal customer segment.")

    # --- Churn Distribution ---
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    df["Churn"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=["yellow", "black"], ax=ax)
    ax.set_title("Distribution of Churn")
    ax.set_ylabel("")
    st.pyplot(fig)

    # --- Loyal by Near Location ---
    st.subheader("Loyal Customers by Near Location")
    fig, ax = plt.subplots()
    loyal["Near_Location"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=["#FFD60A", "#000000"], ax=ax)
    ax.set_title("Loyal Customers by Near Location")
    ax.set_ylabel("")
    st.pyplot(fig)

    # --- Loyal by Age ---
    st.subheader("Age Distribution of Loyal Customers")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(loyal["Age"], bins=20, kde=True, ax=ax, color="#FFB703")
    ax.set_title("Age Distribution (Loyal Customers)")
    st.pyplot(fig)

    # --- Contract vs Lifetime ---
    st.subheader("Contract Duration vs Lifetime (Loyal Customers)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=loyal, x="Contract_period", y="Lifetime", color="#FFD60A", ax=ax)
    ax.set_title("Contract Duration vs Lifetime (Loyal Customers)")
    st.pyplot(fig)

    # --- Near Location + Promo + Group Visits ---
    st.subheader("Near Location vs Group Visits and Friend Promo")
    loyal_near_friend_counts = loyal.groupby(['Near_Location', 'Group_visits', 'Promo_friends']).size().reset_index(name='count')
    loyal_near_friend_counts["Group_Promo"] = loyal_near_friend_counts["Group_visits"].astype(str) + "_" + loyal_near_friend_counts["Promo_friends"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=loyal_near_friend_counts, x='Near_Location', y='count', hue='Group_Promo',
                palette=['#606C38', '#283618', '#BC6C25', '#DDA15E'], ax=ax)
    ax.set_title("Customers by Near_Location, Group Visits, and Promo Friends")
    st.pyplot(fig)

    # --- Lifetime Distribution ---
    st.subheader("Lifetime Distribution (Loyal Customers)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(loyal['Lifetime'], bins=20, kde=False, color='yellow', ax=ax)
    sns.kdeplot(loyal['Lifetime'], bw_method='scott', linewidth=1.5, color='black', ax=ax)
    ax.set_title("Lifetime Distribution of Loyal Customers")
    st.pyplot(fig)

    # --- Age vs Lifetime ---
    st.subheader("Age vs Lifetime (Loyal Customers)")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='Lifetime', data=loyal, ax=ax)
    ax.set_title("Age vs Lifetime (Loyal Customers)")
    st.pyplot(fig)

    # --- Class Frequency Total ---
    st.subheader("Average Class Frequency (Total)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(x='Avg_class_frequency_total', data=loyal, color='yellow', ax=ax)
    ax.set_title("Distribution of Class Frequency (Total)")
    st.pyplot(fig)

    # --- Additional Charges ---
    st.subheader("Additional Charges Distribution")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.histplot(loyal["Avg_additional_charges_total"], bins=20, ax=ax, color="black")
    ax.set_title("Distribution of Additional Spending (Loyal Customers)")
    st.pyplot(fig)

# ============== TAB 2: PREDICTION ==============
with tab2:
    st.header("🔮 Predict Customer Churn")
    st.write("Enter the details below to estimate churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        near_location = st.selectbox("Lives near the gym?", ["Yes", "No"])
        partner = st.selectbox("Has a workout partner?", ["Yes", "No"])
        promo_friends = st.selectbox("Joined through friend’s promotion?", ["Yes", "No"])
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

    # Prepare input dataframe
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
        'Avg_class_frequency_current_month': [avg_class_frequency_current_month]
    })

    # One-hot encode categorical
    categorical_cols = ['gender', 'Near_Location', 'Partner', 'Promo_friends', 'Phone', 'Group_visits']
    raw_dummies = pd.get_dummies(raw_input[categorical_cols])

    num_cols = [
        'Contract_period', 'Age', 'Avg_additional_charges_total',
        'Month_to_end_contract', 'Lifetime',
        'Avg_class_frequency_total', 'Avg_class_frequency_current_month'
    ]
    scaled_num = pd.DataFrame(scaler.transform(raw_input[num_cols]), columns=num_cols, index=raw_input.index)
    prepared = pd.concat([raw_dummies, scaled_num], axis=1)

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        prepared = prepared.reindex(columns=expected_features, fill_value=0)

    if st.button("Predict Churn"):
        pred = model.predict(prepared)[0]
        prob = model.predict_proba(prepared)[0][1]
        if pred == 1:
            st.error(f"⚠️ Customer is **likely to churn**, probability: {prob:.2f}")
        else:
            st.success(f"✅ Customer is **likely to stay**, churn probability: {prob:.2f}")

st.markdown("---")
st.markdown("Built with ❤️ by **Zinedine Amalia** — *Gym Churn Prediction Dashboard*")
