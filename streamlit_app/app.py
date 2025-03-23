# streamlit_app/app.py

import streamlit as st
import numpy as np
import joblib
import os

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "credit_risk_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
thresholds = joblib.load(os.path.join(BASE_DIR, "models", "risk_thresholds.pkl"))

# Set up page configuration
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")

# 1. Branded Header
st.markdown("""
    <style>
        .main-header {
            background-color: #004080;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .main-header h1, .main-header h3 {
            color: white;
            margin: 0;
        }
    </style>
    <div class="main-header">
        <h1>💳 Credit Risk Scoring Portal</h1>
        <h3>Assess Loan Default Risk by Borrower Persona</h3>
    </div>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("📋 About")
st.sidebar.info("This app predicts the credit risk level of a loan applicant using a trained machine learning model.")
st.sidebar.write("Version 2.0")

# 3. Input Form
st.markdown("### 🧾 Applicant Details")

with st.form("risk_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        income = st.number_input("Annual Income ($)", 1000, 200000, 60000)
        credit_score = st.number_input("Credit Score", 300, 850, 650)

    with col2:
        loan_amount = st.number_input("Loan Amount ($)", 500, 100000, 15000)
        months_employed = st.number_input("Months Employed", 0, 480, 24)
        num_credit_lines = st.number_input("Number of Credit Lines", 0, 20, 4)

    with col3:
        interest_rate = st.slider("Interest Rate (%)", 1.0, 40.0, 12.5)
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
        dti = st.slider("Debt-to-Income Ratio", 0.0, 1.5, 0.35)

    st.divider()

    col4, col5, col6 = st.columns(3)

    with col4:
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])

    with col5:
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        mortgage = st.radio("Has Mortgage?", ["Yes", "No"])

    with col6:
        dependents = st.radio("Has Dependents?", ["Yes", "No"])
        purpose = st.selectbox("Loan Purpose", ["Business", "Education", "Home", "Other"])
        co_signer = st.radio("Has Co-Signer?", ["Yes", "No"])

    submitted = st.form_submit_button("🔎 Assess Risk")

# 4. Prediction Logic
if submitted:
    # Numeric Inputs
    numeric = [
        age, income, loan_amount, credit_score, months_employed, num_credit_lines,
        interest_rate, loan_term, dti
    ]

    # Engineered Features
    loan_to_income = loan_amount / (income + 1)
    emp_duration_ratio = months_employed / (age * 12 + 1)

    # Final numeric vector (11 features)
    numeric_full = numeric + [loan_to_income, emp_duration_ratio]
    numeric_scaled = scaler.transform([numeric_full])

    # Encoded Features
    education_map = {"High School": [1, 0, 0], "Bachelor's": [0, 1, 0], "Master's": [0, 0, 1], "PhD": [0, 0, 0]}
    employment_map = {
        "Full-time": [0, 0, 0], "Part-time": [1, 0, 0],
        "Self-employed": [0, 1, 0], "Unemployed": [0, 0, 1]
    }
    marital_map = {"Single": [1, 0], "Married": [0, 1], "Divorced": [0, 0]}
    purpose_map = {
        "Business": [1, 0, 0, 0], "Education": [0, 1, 0, 0],
        "Home": [0, 0, 1, 0], "Other": [0, 0, 0, 1]
    }

    encoded = (
        education_map[education] +
        employment_map[employment] +
        marital_map[marital] +
        [int(mortgage == "Yes")] +
        [int(dependents == "Yes")] +
        purpose_map[purpose] +
        [int(co_signer == "Yes")]
    )

    # Final input for prediction
    final_input = np.concatenate([numeric_scaled[0], np.array(encoded)]).reshape(1, -1)

    # Prediction
    risk_prob = model.predict_proba(final_input)[0][1]
    risk_score = round(risk_prob * 100, 2)

    # Risk category
    if risk_prob < thresholds["low"]:
        risk_level = "🟢 Low Risk"
    elif thresholds["low"] <= risk_prob < thresholds["medium"]:
        risk_level = "🟡 Medium Risk"
    elif thresholds["medium"] <= risk_prob < thresholds["high"]:
        risk_level = "🟠 High Risk"
    else:
        risk_level = "🔴 Extremely High Risk"

    # Display Results
    st.markdown("### 🎯 Risk Prediction")
    st.metric(label="Default Probability", value=f"{risk_score}%")
    st.markdown(f"### 📊 Risk Classification: **{risk_level}**")

    with st.expander("🧾 Explanation"):
        st.write("This classification is based on financial attributes including loan size, income, credit history, and employment duration.")

# 5. Footer
st.markdown("""
---
<center>
    <sub>© 2025 CreditRiskML • Built with ❤️ using Streamlit</sub>
</center>
""", unsafe_allow_html=True)
