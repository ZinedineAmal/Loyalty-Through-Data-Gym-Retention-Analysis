import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import altair as alt

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#FFD60A", "#000000", "#003566"]

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
    st.markdown("Visual insights of **loyal gym members** using the color palette 💛💙⚫.")

    # Define color palette
    gold = "#FFD60A"
    black = "#000000"
    blue = "#003566"

    # Row 1: Churn Distribution & Near Location
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig1 = px.pie(
            churn_counts,
            names="Churn",
            values="Count",
            color_discrete_sequence=[gold, black],
            hole=0.3
        )
        fig1.update_traces(textinfo="percent+label")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Most customers remain loyal (Churn = 0).")

    with col2:
        st.subheader("Loyal Customers by Near Location")
        near_counts = loyal["Near_Location"].value_counts().reset_index()
        near_counts.columns = ["Near_Location", "Count"]
        fig2 = px.bar(
            near_counts,
            x="Near_Location",
            y="Count",
            color_discrete_sequence=[blue]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Loyal members mostly live near the gym.")

    # Row 2: Age Distribution & Contract vs Lifetime
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age Distribution (Loyal Members)")
        fig3 = px.histogram(
            loyal,
            x="Age",
            nbins=20,
            color_discrete_sequence=[gold]
        )
        fig3.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Most loyal members are young adults.")

    with col4:
        st.subheader("Average Lifetime by Contract Period")
        if {"Contract_period", "Lifetime"}.issubset(loyal.columns):
            contract_summary = loyal.groupby("Contract_period")["Lifetime"].mean().reset_index()
            fig4 = px.bar(
                contract_summary,
                x="Contract_period",
                y="Lifetime",
                color_discrete_sequence=[blue]
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Longer contracts tend to yield higher lifetimes.")
        else:
            st.info("Missing Contract_period or Lifetime columns.")

    # Row 3: Group visits & Promo friends
    st.subheader("Group Visits & Friend Promotions (Loyal Members)")
    if {"Group_visits", "Promo_friends"}.issubset(loyal.columns):
        loyal_group = loyal.groupby(["Group_visits", "Promo_friends"]).size().reset_index(name="Count")
        fig5 = px.bar(
            loyal_group,
            x="Group_visits",
            y="Count",
            color="Promo_friends",
            color_discrete_sequence=[black, gold],
            barmode="group",
            labels={"Group_visits": "Group Visits (0=No,1=Yes)", "Promo_friends": "Promo Friends (0=No,1=Yes)"}
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Combining group classes and friend promos enhances loyalty.")
    else:
        st.info("Group_visits or Promo_friends column missing.")

    # Row 4: Lifetime & Additional Charges
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Lifetime Distribution (Loyal Members)")
        if "Lifetime" in loyal.columns:
            fig6 = px.histogram(
                loyal,
                x="Lifetime",
                nbins=20,
                color_discrete_sequence=[blue]
            )
            st.plotly_chart(fig6, use_container_width=True)
            st.caption("Loyal members often stay around 10–20 months.")
        else:
            st.info("Lifetime column missing.")

    with col6:
        st.subheader("Additional Spending (Loyal Members)")
        if "Avg_additional_charges_total" in loyal.columns:
            fig7 = px.histogram(
                loyal,
                x="Avg_additional_charges_total",
                nbins=20,
                color_discrete_sequence=[gold]
            )
            st.plotly_chart(fig7, use_container_width=True)
            st.caption("Loyal members spend more on add-ons like classes or drinks.")
        else:
            st.info("Avg_additional_charges_total column missing.")

    # Final summary
    st.markdown("---")
    st.subheader("📈 Summary Insights")
    st.markdown(f"""
    - 💛 **Location proximity** is a strong loyalty driver.  
    - 💙 **Social engagement** (group classes & friend promos) strengthens retention.  
    - ⚫ **Longer contracts** encourage sustained membership.  
    - 💛 **High spending** members tend to stay longer and show greater engagement.  
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
