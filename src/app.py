import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Assessment", page_icon="💳", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
model_dir = BASE_DIR / "models" / "credit_score_model.pkl"

# Twitch palette
PURPLE = "#9147ff"
SURFACE = "#18181b"
TEXT = "#efeff1"
MUTED = "#adadb8"
RED = "#eb0400"
GREEN = "#00f593"

FEATURE_LABELS = {
    'AMT_CREDIT': "Credit amount",
    'AMT_ANNUITY': "Annual installment",
    'YEARS_EMPLOYED_CLEAN': "Years employed",
    'INCOME_PER_CHILD': "Income per child",
    'PAYMENT_RATE': "Installment / income",
    'CREDIT_GOODS_RATIO': "Credit / goods price",
    'CREDIT_DOWNPAYMENT': "Down payment",
    'EXT_SOURCE_1': "Bureau score 1",
    'EXT_SOURCE_2': "Bureau score 2",
    'EXT_SOURCE_3': "Bureau score 3",
    'REGION_RATING_CLIENT': "Region rating",
    'IS_RETIRED': "Retired",
}

@st.cache_resource
def load_model():
    return joblib.load(str(model_dir))


@st.cache_resource
def get_tree_explainer():
    return shap.TreeExplainer(load_model())


def styled_barh(series, colors):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(series) + 0.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(series.index, series.values, color=colors)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0, color="#2f2f35", linewidth=1)
    fig.tight_layout()
    return fig


try:
    model = load_model()
except Exception as e:
    st.error(f"Actual Error: {e}")
    st.write(f"Model file 'credit_score_model.pkl' not found. model_dir: **{model_dir}**")
    st.stop()

# ---------------------------------------------------------------

region_options = {
    "Level 1: High-income/Low-risk Region": 1,
    "Level 2: Average Region": 2,
    "Level 3: Low-income/High-risk Region": 3
}

with st.sidebar:
    st.header("Applicant")

    st.subheader("Loan")
    # Defaults are the medians of application_train.csv
    credit = st.number_input("Total credit amount ($)", min_value=1, value=513500, step=10000)
    goods_price = st.number_input("Price of the goods ($)", min_value=1, value=450000, step=10000,
                                  help="Price of the item the loan is paying for.")
    annuity = st.number_input("Annual loan installment ($)", min_value=0, value=24900, step=1000)

    st.subheader("Income & work")
    income = st.number_input("Total annual income ($)", min_value=1, value=147150, step=10000)
    children = st.number_input("Number of children", min_value=0, value=0)
    status = st.segmented_control("Employment status", ["Employed", "Retired"], default="Employed")
    retired = int(status == "Retired")
    # In training, retired applicants always have 0 years employed
    years_employed = 0 if retired else st.number_input("Years employed", min_value=0, value=4)
    selected_region = st.selectbox("Region rating", options=list(region_options.keys()), index=1)

    st.subheader("Credit bureau scores")
    st.caption("Normalized scores from external bureaus. Higher means more trustworthy.")
    ext_1 = st.slider("Bureau score 1", 0.0, 1.0, 0.51)
    ext_2 = st.slider("Bureau score 2", 0.0, 1.0, 0.57)
    ext_3 = st.slider("Bureau score 3", 0.0, 1.0, 0.54)

# ---------------------------------------------------------------

income_per_child = income / (children + 1)
payment_rate = annuity / income
credit_goods_ratio = credit / goods_price
credit_downpayment = goods_price - credit
region_rating = region_options[selected_region]


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

st.title("Credit Risk Assessment")
st.caption("Fill in the applicant's details on the left. The score updates automatically.")

# Get Probability
prob = model.predict_proba(input_df)[0, 1]

max_observed_prob = 0.9614
min_observed_prob = 0.0085

# Normalization formula:
# We map the range [0.0085, 0.9614] to [1000, 0]
adjusted_score = 1000 * (1 - (prob - min_observed_prob) / (max_observed_prob - min_observed_prob))
credit_score = int(max(0, min(1000, adjusted_score)))

# The model is trained with scale_pos_weight, so its probabilities are inflated.
# Platt scaling on the logit, fitted in the notebook on a held-out half of the test set.
calib_a, calib_b = 0.9975, -2.3828
default_prob = 1 / (1 + np.exp(-(calib_a * np.log(prob / (1 - prob)) + calib_b)))

explainer = get_tree_explainer()
shap_values = explainer.shap_values(input_df)
if isinstance(shap_values, list):
    shap_row = shap_values[1][0]
else:
    shap_row = shap_values[0]

# Display Results
col1, col2, col3 = st.columns(3)
col1.metric("Credit score (out of 1000)", credit_score, border=True)
col2.metric("Probability of default", f"{default_prob:.1%}", border=True,
            help="Calibrated estimate. The average applicant in the dataset defaults 8.1% of the time.")
with col3.container(border=True, height="stretch"):
    st.caption("Decision")
    if credit_score >= 700:
        st.badge("Approved · Low risk", icon=":material/check_circle:", color="green")
    elif credit_score >= 400:
        st.badge("Review · Medium risk", icon=":material/visibility:", color="orange")
    else:
        st.badge("Rejected · High risk", icon=":material/cancel:", color="red")

st.progress(credit_score / 1000, text="0 · Rejected < 400 ≤ Review < 700 ≤ Approved · 1000")

contrib = pd.Series(shap_row, index=[FEATURE_LABELS[c] for c in input_df.columns])
raising = contrib[contrib > 0].sort_values(ascending=False).head(3)
lowering = contrib[contrib < 0].sort_values().head(3)

st.subheader("Why this score?")
left, right = st.columns([3, 2])
with left:
    top = contrib.reindex(contrib.abs().sort_values().index).tail(8)
    st.pyplot(styled_barh(top, [RED if v > 0 else GREEN for v in top]))
    st.caption("Red bars push default risk up, green bars push it down.")
with right:
    st.markdown("**Raising the risk**")
    st.markdown("\n".join(f"- {name}" for name in raising.index) or "Nothing significant.")
    st.markdown("**Lowering the risk**")
    st.markdown("\n".join(f"- {name}" for name in lowering.index) or "Nothing significant.")

with st.expander("How the model works"):
    st.markdown(
        "An **XGBoost** model estimates the probability that the applicant defaults, "
        "which is then mapped to a **0–1000 credit score**. "
        "The chart shows which inputs the model relies on most overall."
    )
    importances = pd.Series(model.feature_importances_, index=[FEATURE_LABELS[c] for c in input_df.columns])
    st.pyplot(styled_barh(importances.sort_values(), PURPLE))
