import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import altair as alt

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
import plotly.express as px
import altair as alt

with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Interactive visual analysis of **loyal gym members**.")

    # ==========================
    # Layout 1: Churn & Near Location
    # ==========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(churn_counts, names="Churn", values="Count",
                     color_discrete_sequence=["gold", "black"],
                     hole=0.3)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Most customers remain **loyal** (did not churn).")

    with col2:
        st.subheader("Loyal Customers by Near Location")
        near_counts = loyal["Near_Location"].value_counts().reset_index()
        near_counts.columns = ["Near_Location", "Count"]
        st.bar_chart(near_counts.set_index("Near_Location"))
        st.caption("Loyal members mostly live **near the gym**, showing proximity helps retention.")

    # ==========================
    # Layout 2: Age & Contract
    # ==========================
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Age Distribution (Loyal Members)")
        age_hist = np.histogram(loyal["Age"], bins=20)[0]
        st.bar_chart(age_hist)
        st.caption("Most loyal members are between **25–35 years old**.")

    with col4:
        st.subheader("Contract Duration vs Lifetime")
        contract_summary = loyal.groupby("Contract_period")["Lifetime"].mean().reset_index()
        contract_chart = alt.Chart(contract_summary).mark_bar(color="gold").encode(
            x="Contract_period:O",
            y="Lifetime:Q",
            tooltip=["Contract_period", "Lifetime"]
        )
        st.altair_chart(contract_chart, use_container_width=True)
        st.caption("Members with **longer contracts** tend to stay longer (higher lifetime).")

    # ==========================
    # Layout 3: Group Visits + Friend Promo
    # ==========================
    st.subheader("Group Visits and Friend Promotions (Loyal Members)")
    loyal_group = loyal.groupby(["Group_visits", "Promo_friends"]).size().reset_index(name="Count")
    fig = px.bar(
        loyal_group, x="Group_visits", y="Count", color="Promo_friends",
        barmode="group", color_discrete_sequence=["#606C38", "#DDA15E"]
    )
    fig.update_layout(xaxis_title="Group Visits (0=No, 1=Yes)", yaxis_title="Customer Count")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Combining **group classes** and **friend promos** correlates with the highest loyalty.")

    # ==========================
    # Layout 4: Lifetime & Spending
    # ==========================
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Lifetime Distribution")
        fig = px.histogram(loyal, x="Lifetime", nbins=20, color_discrete_sequence=["#FFD60A"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Most loyal customers stay around **10–20 months** on average.")

    with col6:
        st.subheader("Additional Spending")
        fig = px.histogram(loyal, x="Avg_additional_charges_total", nbins=20,
                           color_discrete_sequence=["black"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Loyal members tend to **spend more** on add-ons like training and products.")

    # ==========================
    # Layout 5: Attendance Frequency
    # ==========================
    st.subheader("Class Attendance Frequency (Total)")
    freq_chart = px.histogram(loyal, x="Avg_class_frequency_total",
                              color_discrete_sequence=["#FFB703"])
    st.plotly_chart(freq_chart, use_container_width=True)
    st.caption("Customers who attend **more than 3 sessions per week** are the most loyal.")

    # ==========================
    # Summary Insight
    # ==========================
    st.markdown("---")
    st.subheader("📈 Summary of Key Insights")
    st.markdown("""
    - 🏠 **Location matters:** Members living near the gym show higher loyalty.
    - 🤝 **Social engagement helps:** Group classes and friend promos boost retention.
    - 📅 **Longer contracts = longer lifetime.**
    - 💰 **High-spending and frequent attendees** are the most loyal segment.
    - 👥 **Young adults (25–35)** dominate the loyal customer base.
    """)


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
