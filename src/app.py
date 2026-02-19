import streamlit as st
import joblib
import  pandas as pd
import matplotlib.pyplot as plt
import os

st.title("Credit Risk Assessment")
st.markdown("Defaulting Risk evaluation based on smart features.")

model_dir = "models/credit_score_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(model_dir)

try:
    model = load_model()
except:
    st.error(f"Model file 'credit_score_model.pkl' not found. model_dir: **{model_dir}**")

# ---------------------------------------------------------------

region_options = {
    "Level 1: High-income/Low-risk Region": 1,
    "Level 2: Average Region": 2,
    "Level 3: Low-income/High-risk Region": 3
}

retired_options = {
    "Yes": 1,
    "No": 0
}

st.sidebar.header("Applicant Information")


credit = st.sidebar.number_input("Total Credit Amount ($)", value=15000, step=1000)
annuity = st.sidebar.number_input("Annual Loan Installment ($)", value=5000, step=500)
years_employed = st.sidebar.number_input("Years Employed", value=5)
children = st.sidebar.number_input("Number of Children", value=0)
income = st.sidebar.number_input("Total Annual Income ($)", value=50000, step=1000)
goods_price = st.sidebar.number_input("Goods Price ($)", value=15000, step=1000)
selected_region = st.sidebar.selectbox("Region Rating", options=list(region_options.keys()))
retired_value = st.sidebar.selectbox("Are you retired?", options=list(retired_options.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("External Credit Scores")
ext_1 = st.sidebar.slider("External Source 1", 0.0, 1.0, 0.5)
ext_2 = st.sidebar.slider("External Source 2", 0.0, 1.0, 0.5)
ext_3 = st.sidebar.slider("External Source 3", 0.0, 1.0, 0.5)

# ---------------------------------------------------------------

income_per_child = income / (children + 1)
payment_rate = annuity / income
credit_goods_ratio = credit / goods_price
credit_downpayment = goods_price - credit
region_rating = region_options[selected_region]
retired = retired_options[retired_value]


# ---------------------------------------------------------------
# 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE', 'YEARS_EMPLOYED_CLEAN', 
# 'INCOME_PER_CHILD', 'PAYMENT_RATE', 'CREDIT_GOODS_RATIO', 'CREDIT_DOWNPAYMENT'
# 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'REGION_RATING_CLIENT'
input_df = pd.DataFrame([[
    credit, 
    annuity, 
    years_employed,
    income_per_child,
    payment_rate,
    credit_goods_ratio, 
    credit_downpayment,
    ext_1, 
    ext_2,
    ext_3,
    region_rating,
    retired
]], columns=[
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'YEARS_EMPLOYED_CLEAN',
    'INCOME_PER_CHILD',
    'PAYMENT_RATE',
    'CREDIT_GOODS_RATIO',
    'CREDIT_DOWNPAYMENT',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'REGION_RATING_CLIENT',
    'IS_RETIRED'
])

# ---------------------------------------------------------------

st.markdown("""
This dashboard uses an **XGBoost** model to predict the probability of default. 
It converts that probability into a **Credit Score** (0-1000).
""")

if st.button("Calculate Credit Score"):
    # Get Probability
    prob = model.predict_proba(input_df)[0, 1]
    
    max_observed_prob = 0.9640
    min_observed_prob = 0.01

    # Normalization formula:
    # We map the range [0.01, 0.45] to [1000, 0]
    adjusted_score = 1000 * (1 - (prob - min_observed_prob) / (max_observed_prob - min_observed_prob))
    credit_score = int(max(0, min(1000, adjusted_score)))



    # Display Results
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Calculated Credit Score", value=f"{credit_score} / 1000")
        
        if credit_score >= 700:
            st.success("Decision: **APPROVED** (Low Risk)")
        elif credit_score >= 400:
            st.warning("Decision: **MANUAL REVIEW REQUIRED** (Medium Risk)")
        else:
            st.error("Decision: **REJECTED** (High Risk)")

    with col2:
        # Local Insights
        st.write("### Key Risk Factors")
        if payment_rate > 0.4:
            st.write("**High Debt-to-Income:** Installment exceeds 40% of income.")
        if ext_3 < 0.3:
            st.write("**Low External Rating:** Third-party bureaus report high risk.")
        if credit_score > 500:
            st.write("**Solid Applicant Profile:** Financial indicators are stable.")

# 5. Feature Importance Visualization
    st.divider()
    st.subheader("Model Decision Drivers (Global Importance)")
    importances = pd.Series(model.feature_importances_, index=input_df.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind='barh', ax=ax, color='#004481') # BBVA Blue
    st.pyplot(fig)