import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Gym Churn Dashboard", layout="wide")

# === HEADER IMAGE (centered banner) ===
st.markdown(
    """
    <style>
    .gym {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;            /* ubah 60–90% kalau mau lebih kecil/besar */
        max-height: 300px;     /* batasi tinggi banner */
        object-fit: cover;     /* biar tidak ketarik */
        border-radius: 15px;   /* sudut melengkung agar elegan */
    }
    </style>
    <img src="gym.jpg" class="gym">
    """,
    unsafe_allow_html=True
)


# === TITLE ===
st.markdown("<h1 style='text-align: center;'>🏋️‍♀️ Gym Customer Churn Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Explore customer insights and predict churn probabilities.")

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
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(int)
    else:
        st.error("Column 'Churn' not found in dataset.")
        st.stop()
    loyal = df[df["Churn"] == 0].copy()
    return df, loyal

df, loyal = load_data()

# ============== TAB STRUCTURE ==============
tab1, tab2 = st.tabs(["🔮 Churn Prediction", "📊 Exploratory Data Analysis (EDA)"])

# ============== TAB 1: PREDICTION ==============
with tab1:
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

# ============== TAB 2: EDA ==============
with tab2:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Visual insights of **loyal gym members** using the color palette 💛💙⚫.")

    gold = "#FFD60A"
    black = "#000000"
    blue = "#003566"

    # === Row 1 ===
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
            color_discrete_sequence=[black]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Loyal members mostly live near the gym.")

    # === Row 2 ===
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

    # === New Chart: Age vs Lifetime ===
    st.subheader("Average Lifetime by Age")
    if {"Age", "Lifetime"}.issubset(loyal.columns):
        age_life = loyal.groupby("Age")["Lifetime"].mean().reset_index()
        fig8 = px.line(
            age_life,
            x="Age",
            y="Lifetime",
            color_discrete_sequence=[gold],
            markers=True
        )
        fig8.update_layout(xaxis_title="Age", yaxis_title="Average Lifetime (months)")
        st.plotly_chart(fig8, use_container_width=True)
        st.caption("Members aged 25–35 tend to have the highest lifetime duration.")
    else:
        st.info("Age or Lifetime column missing.")

    # === New Chart 2: Lifetime vs Additional Spending ===
    st.subheader("Lifetime vs Additional Spending by Location")
    if {"Lifetime", "Avg_additional_charges_total", "Near_Location"}.issubset(loyal.columns):
        fig9 = px.scatter(
            loyal,
            x="Lifetime",
            y="Avg_additional_charges_total",
            color="Near_Location",
            color_discrete_sequence=[blue, gold],
            trendline="ols",
            opacity=0.7
        )
        fig9.update_layout(
            xaxis_title="Lifetime (months)",
            yaxis_title="Avg Additional Charges (USD)"
        )
        st.plotly_chart(fig9, use_container_width=True)
        st.caption("Nearby customers tend to stay longer and spend more.")
    else:
        st.info("Lifetime, Avg_additional_charges_total, or Near_Location column missing.")

    # === Row 3 ===
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

    # === Row 4 ===
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

    # === UPDATED FINAL SUMMARY ===
    st.markdown("---")
    st.subheader("📈 Summary Insights")
    st.markdown("""
    - 💛 **Customers aged 25–35** have the highest lifetime and loyalty potential.  
    - 💙 **Most loyal customers live near the gym** and often have a partner.  
    - ⚫ **Friend referral promos** work best for customers in nearby areas.  
    - 💛 **Average visits:** 1–4 times per month.  
    - 💙 **Average additional spending:** $50–$200.  
    - ⚫ **As lifetime increases**, class participation and spending tend to decline.  
    """)

# ============== AUTHOR SECTION ==============
st.markdown("---")
st.markdown("""
## 👩‍💻 Author
**Zinedine Amalia**  
_Data Analyst | Customer Insights & Retention Strategy_

## 🤝 Connect with Me
<p align="center">
  <a href="https://linkedin.com/in/YOUR-zinedineamalia/" target="_blank">
    <img src="https://img.icons8.com/color/48/linkedin.png" alt="LinkedIn"/>
  </a>
  <a href="mailto:zinedineamalianoor.com">
    <img src="https://img.icons8.com/color/48/gmail-new.png" alt="Gmail"/>
  </a>
  <a href="https://instagram.com/z.amalia_" target="_blank">
    <img src="https://img.icons8.com/color/48/instagram-new.png" alt="Instagram"/>
  </a>
</p>
""", unsafe_allow_html=True)
