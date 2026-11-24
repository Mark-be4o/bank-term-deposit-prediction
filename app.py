import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered",
)

# ----------------------------------------------------------------------------
# Load model + preprocessor (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("Model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["preprocessor"], artifacts["model"]


preprocessor, model = load_artifacts()

NUM_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CAT_COLS = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome",
]

# Categories pulled directly from the trained OneHotEncoder so the dropdowns
# always match what the model was trained on.
cat_encoder = preprocessor.named_transformers_["cat"]
CAT_OPTIONS = dict(zip(CAT_COLS, [list(c) for c in cat_encoder.categories_]))

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🏦 Bank Marketing Campaign Predictor")
st.write(
    "Predict whether a client will subscribe to a term deposit, based on the "
    "bank's marketing campaign data. Fill in the client's details below and "
    "click **Predict**."
)

st.divider()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("client_form"):
    st.subheader("Client Information")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
        job = st.selectbox("Job", CAT_OPTIONS["job"])
        marital = st.selectbox("Marital status", CAT_OPTIONS["marital"])
        education = st.selectbox("Education", CAT_OPTIONS["education"])
        default = st.selectbox("Has credit in default?", CAT_OPTIONS["default"])
    with c2:
        housing = st.selectbox("Has housing loan?", CAT_OPTIONS["housing"])
        loan = st.selectbox("Has personal loan?", CAT_OPTIONS["loan"])
        contact = st.selectbox("Contact communication type", CAT_OPTIONS["contact"])
        balance = st.number_input("Average yearly balance (€)", value=1000, step=100)

    st.subheader("Campaign Information")
    c3, c4 = st.columns(2)
    with c3:
        month = st.selectbox("Last contact month", CAT_OPTIONS["month"])
        day = st.number_input("Last contact day of month", min_value=1, max_value=31, value=15, step=1)
        duration = st.number_input("Last contact duration (seconds)", min_value=0, value=180, step=10)
        campaign = st.number_input("Number of contacts this campaign", min_value=1, value=1, step=1)
    with c4:
        pdays = st.number_input(
            "Days since last contact from a previous campaign (-1 = never contacted)",
            min_value=-1, value=-1, step=1,
        )
        previous = st.number_input("Number of contacts before this campaign", min_value=0, value=0, step=1)
        poutcome = st.selectbox("Outcome of previous campaign", CAT_OPTIONS["poutcome"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    input_dict = {
        "age": age, "balance": balance, "day": day, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan,
        "contact": contact, "month": month, "poutcome": poutcome,
    }
    input_df = pd.DataFrame([input_dict])[NUM_COLS + CAT_COLS]

    X_processed = preprocessor.transform(input_df)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    prob = float(model.predict(X_processed, verbose=0)[0][0])
    prediction = "Yes" if prob >= 0.5 else "No"

    st.divider()
    st.subheader("Prediction Result")

    if prediction == "Yes":
        st.success(f"✅ The client is likely to **subscribe** to a term deposit.")
    else:
        st.error(f"❌ The client is **unlikely** to subscribe to a term deposit.")

    st.metric("Subscription probability", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    with st.expander("See input data sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Model: Keras neural network (128-64-32-1) with scikit-learn preprocessing pipeline.")
