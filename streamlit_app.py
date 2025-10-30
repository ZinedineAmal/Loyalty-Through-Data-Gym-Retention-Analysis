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
    # Ensure the Churn column exists and has correct dtype
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(int)
    else:
        st.error("Column 'Churn' not found in dataset.")
        st.stop()

    # Create loyal subset (Churn == 0)
    loyal = df[df["Churn"] == 0].copy()
    return df, loyal

# call it once
df, loyal = load_data()


# ============== TAB STRUCTURE ==============
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🔮 Churn Prediction"])

# ============== TAB 1: EDA ==============
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Interactive visual analysis of **loyal gym members**.")

    # Row 1: Churn pie & Near Location bar
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(churn_counts, names="Churn", values="Count",
                     color_discrete_sequence=["gold", "black"], hole=0.3)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Most customers remain loyal (Churn = 0).")

    with col2:
        st.subheader("Loyal Customers by Near Location")
        near_counts = loyal["Near_Location"].value_counts().reset_index()
        near_counts.columns = ["Near_Location", "Count"]
        fig2 = px.bar(near_counts, x="Near_Location", y="Count", labels={"Near_Location": "Near Location"})
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Loyal members mostly live near the gym.")

    # Row 2: Age distribution & Contract vs Lifetime
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age Distribution (Loyal Members)")
        # use plotly histogram directly from the DataFrame — no np.histogram needed
        fig3 = px.histogram(loyal, x="Age", nbins=20, color_discrete_sequence=["#FFB703"])
        fig3.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Most loyal members appear in younger adult groups.")

    with col4:
        st.subheader("Average Lifetime by Contract Period")
        if "Contract_period" in loyal.columns and "Lifetime" in loyal.columns:
            contract_summary = loyal.groupby("Contract_period")["Lifetime"].mean().reset_index()
            fig4 = px.bar(contract_summary, x="Contract_period", y="Lifetime",
                          labels={"Contract_period": "Contract Period (months)", "Lifetime": "Avg Lifetime (months)"},
                          color="Lifetime", color_continuous_scale="sunset")
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Longer contract periods show higher average lifetimes.")
        else:
            st.info("Contract_period or Lifetime column missing from dataset.")

    # Row 3: Group visits & Promo friends
    st.subheader("Group Visits & Friend Promotions (Loyal Members)")
    if {"Group_visits", "Promo_friends"}.issubset(loyal.columns):
        loyal_group = loyal.groupby(["Group_visits", "Promo_friends"]).size().reset_index(name="Count")
        fig5 = px.bar(loyal_group, x="Group_visits", y="Count", color="Promo_friends", barmode="group",
                      labels={"Group_visits": "Group Visits (0=No,1=Yes)", "Promo_friends": "Promo Friends (0=No,1=Yes)"})
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Group classes combined with friend promos are associated with higher loyalty.")
    else:
        st.info("Group_visits or Promo_friends column missing from dataset.")

    # Row 4: Lifetime and Spending
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Lifetime Distribution (Loyal)")
        if "Lifetime" in loyal.columns:
            fig6 = px.histogram(loyal, x="Lifetime", nbins=20, color_discrete_sequence=["#FFD60A"])
            st.plotly_chart(fig6, use_container_width=True)
            st.caption("Most loyal customers stay around 10–20 months.")
        else:
            st.info("Lifetime column missing.")

    with col6:
        st.subheader("Additional Charges Distribution")
        if "Avg_additional_charges_total" in loyal.columns:
            fig7 = px.histogram(loyal, x="Avg_additional_charges_total", nbins=20, color_discrete_sequence=["black"])
            st.plotly_chart(fig7, use_container_width=True)
            st.caption("Loyal members tend to spend more on add-ons.")
        else:
            st.info("Avg_additional_charges_total column missing.")

    # Summary
    st.markdown("---")
    st.subheader("📈 Summary of Key Insights")
    st.markdown("""
    - **Location matters:** Members living near the gym show higher loyalty.  
    - **Social engagement helps:** Group classes and friend promotions boost retention.  
    - **Contract length matters:** Longer contracts correlate with longer membership lifetime.  
    - **Engaged spenders:** Loyal members spend more on add-ons and attend more frequently.
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
