import streamlit as st
import numpy as np
from joblib import load

# ---------- Load model & scaler ----------
model = load("best_loan_model.joblib")
scaler = load("scaler.joblib")

st.set_page_config(page_title="Loan Approval App", layout="centered")

st.title("🏦 Loan Approval Prediction")

st.write("Enter applicant details")

# ---------- Inputs ----------
no_of_dependents = st.number_input("No of Dependents", 0, 10, 0)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.number_input("Annual Income", 0)
loan_amount = st.number_input("Loan Amount", 0)
loan_term = st.number_input("Loan Term (months)", 1)

cibil_score = st.number_input("CIBIL Score", 300, 900, 600)

residential_assets_value = st.number_input("Residential Assets Value", 0)
commercial_assets_value = st.number_input("Commercial Assets Value", 0)
luxury_assets_value = st.number_input("Luxury Assets Value", 0)
bank_asset_value = st.number_input("Bank Asset Value", 0)

# ---------- Encoding ----------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# ---------- Prediction ----------
if st.button("Predict Loan Status"):

    input_data = np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], dtype=float)

    st.write("Input shape:", input_data.shape)  # MUST be (1,11)
    st.write("Scaler expects:", scaler.n_features_in_)

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0]
    approval_prob = prob[1]

    st.write("Approval Probability:", round(approval_prob*100,2), "%")

    # Use threshold (because ML not always 100% correct)
    if approval_prob >= 0.40:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")


