import streamlit as st
import joblib
import  pandas as pd
import matplotlib.pyplot as plt
import os

st.title("Credit Risk Calculator")
st.markdown("Defaulting Risk evaluation based on smart features.")

# root_dir = os.path.dirname(os.getcwd()) 
# model_dir = os.path.join(root_dir, "models", "credit_score_model.pkl")
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

st.sidebar.header("Applicant Information")


credit = st.sidebar.number_input("Total Credit Amount ($)", value=15000, step=1000)
annuity = st.sidebar.number_input("Annual Loan Installment ($)", value=5000, step=500)
age = st.sidebar.slider("Age (Years)", 18, 90, 30)
years_employed = st.sidebar.number_input("Years Employed", value=5)
children = st.sidebar.number_input("Number of Children", value=0)
income = st.sidebar.number_input("Total Annual Income ($)", value=50000, step=1000)
goods_price = st.sidebar.number_input("Goods Price ($)", value=15000, step=1000)
selected_region = st.sidebar.selectbox("Region Rating", options=list(region_options.keys()))


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

# ---------------------------------------------------------------

input_df = pd.DataFrame([[
    credit, 
    annuity, 
    age,
    years_employed,
    ext_1, 
    ext_2,
    ext_3,
    income_per_child,
    payment_rate,
    region_rating,
    credit_goods_ratio, 
    credit_downpayment
]], columns=[
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AGE',
    'YEARS_EMPLOYED',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'INCOME_PER_CHILD',
    'PAYMENT_RATE',
    'REGION_RATING_CLIENT',
    'CREDIT_GOODS_RATIO',
    'CREDIT_DOWNPAYMENT'
])

# ---------------------------------------------------------------

st.title("🏦 BBVA Credit Risk Assessment")
st.markdown("""
This dashboard uses an **XGBoost** model to predict the probability of default. 
It converts that probability into a **Credit Score** (0-1000).
""")

if st.button("Calculate Credit Score"):
    # Get Probability
    prob_default = model.predict_proba(input_df)[0, 1]
    
    # Calculate Score
    # Higher Score = Lower Risk
    credit_score = int((1 - prob_default) * 1000)
    
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
            st.write("⚠️ **High Debt-to-Income:** Installment exceeds 40% of income.")
        if ext_3 < 0.3:
            st.write("⚠️ **Low External Rating:** Third-party bureaus report high risk.")
        if credit_score > 500:
            st.write("✅ **Solid Applicant Profile:** Financial indicators are stable.")

# 5. Feature Importance Visualization
    st.divider()
    st.subheader("Model Decision Drivers (Global Importance)")
    importances = pd.Series(model.feature_importances_, index=input_df.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind='barh', ax=ax, color='#004481') # BBVA Blue
    st.pyplot(fig)