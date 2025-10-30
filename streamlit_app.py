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
    st.markdown("Below are visual insights from the **loyal customer segment** — customers who have not churned.")

    # --- Layout 1: Churn Distribution & Near Location ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        df["Churn"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", startangle=90, colors=["#FFD60A", "#000000"], ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** About the overall churn proportion.  
            **Conclusion:** The majority of gym members remain **loyal**, indicating a healthy retention base but still room for improvement.
            """
        )

    with col2:
        st.subheader("Loyal Customers by Near Location")
        fig, ax = plt.subplots()
        loyal["Near_Location"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", startangle=90, colors=["#FFD60A", "#000000"], ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** Distribution of loyal customers who live near vs. far from the gym.  
            **Conclusion:** Most loyal members live **near the gym**, suggesting proximity strongly supports loyalty.
            """
        )

    # --- Layout 2: Age Distribution & Contract vs Lifetime ---
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age Distribution of Loyal Customers")
        fig, ax = plt.subplots()
        sns.histplot(loyal["Age"], bins=20, kde=True, ax=ax, color="#FFB703")
        ax.set_title("")
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** Younger adults dominate the loyal segment.  
            **Conclusion:** Customers aged **25–35** tend to be more active and consistent gym-goers.
            """
        )

    with col4:
        st.subheader("Contract Duration vs Lifetime")
        fig, ax = plt.subplots()
        sns.boxplot(data=loyal, x="Contract_period", y="Lifetime", color="#FFD60A", ax=ax)
        ax.set_title("")
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** Relationship between membership contract length and retention.  
            **Conclusion:** Longer contract periods are associated with **longer lifetime durations**, showing contractual commitment increases loyalty.
            """
        )

    # --- Layout 3: Group Visits & Friend Promo ---
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Near Location, Group Visits, and Friend Promo")
        loyal_near_friend_counts = loyal.groupby(['Near_Location', 'Group_visits', 'Promo_friends']).size().reset_index(name='count')
        loyal_near_friend_counts["Group_Promo"] = loyal_near_friend_counts["Group_visits"].astype(str) + "_" + loyal_near_friend_counts["Promo_friends"].astype(str)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=loyal_near_friend_counts, x='Near_Location', y='count', hue='Group_Promo',
                    palette=['#606C38', '#283618', '#BC6C25', '#DDA15E'], ax=ax)
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** Group classes and friend promotions combined drive engagement.  
            **Conclusion:** Members who both attend **group classes** and join through **friend promos** show the **highest loyalty**.
            """
        )

    with col6:
        st.subheader("Lifetime Distribution (Loyal Customers)")
        fig, ax = plt.subplots()
        sns.histplot(loyal['Lifetime'], bins=20, kde=False, color='yellow', ax=ax)
        sns.kdeplot(loyal['Lifetime'], bw_method='scott', linewidth=1.5, color='black', ax=ax)
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** Distribution of how long loyal customers stay members.  
            **Conclusion:** Most loyal customers have a **membership lifetime of 10–20 months**, with a gradual decline after 24 months.
            """
        )

    # --- Layout 4: Age vs Lifetime & Additional Spending ---
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Age vs Lifetime (Loyal Customers)")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Age', y='Lifetime', data=loyal, ax=ax, color="#023047")
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** Relationship between age and how long customers stay.  
            **Conclusion:** Middle-aged customers (30–40) tend to remain longer, likely due to **stable lifestyle patterns**.
            """
        )

    with col8:
        st.subheader("Additional Charges Distribution")
        fig, ax = plt.subplots()
        sns.histplot(loyal["Avg_additional_charges_total"], bins=20, ax=ax, color="black")
        st.pyplot(fig)
        st.markdown(
            """
            **Insight:** Spending habits among loyal customers.  
            **Conclusion:** Higher spending on add-ons (training, products) is typical among **loyal customers**, indicating engagement and brand trust.
            """
        )

    # --- Layout 5: Class Frequency ---
    st.subheader("Average Class Frequency (Total)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(x='Avg_class_frequency_total', data=loyal, color='yellow', ax=ax)
    st.pyplot(fig)
    st.markdown(
        """
        **Insight:** Frequency of gym attendance among loyal customers.  
        **Conclusion:** Customers who attend more than **3 classes per week** are significantly more loyal.
        """
    )


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
