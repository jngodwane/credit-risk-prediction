import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH


st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("Credit Risk Prediction Dashboard")
st.write("Enter borrower details, generate engineered features, and score default risk.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.sidebar.header("Borrower Inputs")

annual_income = st.sidebar.number_input("Annual Income", min_value=1000.0, value=60000.0, step=1000.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=500.0, value=25000.0, step=500.0)
total_debt = st.sidebar.number_input("Total Debt", min_value=0.0, value=15000.0, step=500.0)
credit_balance = st.sidebar.number_input("Credit Balance", min_value=0.0, value=5000.0, step=500.0)
credit_limit = st.sidebar.number_input("Credit Limit", min_value=1.0, value=10000.0, step=500.0)
num_delinquencies = st.sidebar.number_input("Number of Delinquencies", min_value=0, value=1, step=1)
monthly_payment = st.sidebar.number_input("Monthly Payment", min_value=0.0, value=800.0, step=50.0)
credit_history_length = st.sidebar.number_input("Credit History Length", min_value=1.0, value=5.0, step=1.0)
num_credit_inquiries = st.sidebar.number_input("Credit Inquiries", min_value=0, value=2, step=1)
employment_years = st.sidebar.number_input("Employment Years", min_value=0.0, value=3.0, step=1.0)

monthly_income = annual_income / 12
loan_to_income_ratio = loan_amount / annual_income if annual_income else 0
debt_to_income_ratio = total_debt / annual_income if annual_income else 0
credit_utilization_ratio = credit_balance / credit_limit if credit_limit else 0
payment_to_income_ratio = monthly_payment / monthly_income if monthly_income else 0
delinquency_frequency = num_delinquencies / credit_history_length if credit_history_length else 0
has_delinquency_flag = 1 if num_delinquencies > 0 else 0
credit_inquiry_rate = num_credit_inquiries / credit_history_length if credit_history_length else 0
employment_stability_flag = 1 if employment_years >= 3 else 0

if employment_years <= 1:
    employment_years_bucket = "0-1"
elif employment_years <= 3:
    employment_years_bucket = "1-3"
elif employment_years <= 5:
    employment_years_bucket = "3-5"
elif employment_years <= 10:
    employment_years_bucket = "5-10"
else:
    employment_years_bucket = "10+"

financial_stress_score = (
    0.5 * debt_to_income_ratio +
    0.5 * credit_utilization_ratio
)

behavioral_risk_score = (
    0.4 * debt_to_income_ratio +
    0.3 * credit_utilization_ratio +
    0.3 * delinquency_frequency
)

loan_income_utilization_interaction = (
    loan_to_income_ratio * credit_utilization_ratio
)

affordability_stress_interaction = (
    payment_to_income_ratio * debt_to_income_ratio
)

input_df = pd.DataFrame([{
    "customer_id": 999,
    "annual_income": annual_income,
    "loan_amount": loan_amount,
    "total_debt": total_debt,
    "credit_balance": credit_balance,
    "credit_limit": credit_limit,
    "num_delinquencies": num_delinquencies,
    "monthly_payment": monthly_payment,
    "credit_history_length": credit_history_length,
    "num_credit_inquiries": num_credit_inquiries,
    "employment_years": employment_years,
    "loan_to_income_ratio": loan_to_income_ratio,
    "debt_to_income_ratio": debt_to_income_ratio,
    "credit_utilization_ratio": credit_utilization_ratio,
    "payment_to_income_ratio": payment_to_income_ratio,
    "delinquency_frequency": delinquency_frequency,
    "has_delinquency_flag": has_delinquency_flag,
    "credit_inquiry_rate": credit_inquiry_rate,
    "employment_stability_flag": employment_stability_flag,
    "employment_years_bucket": employment_years_bucket,
    "financial_stress_score": financial_stress_score,
    "behavioral_risk_score": behavioral_risk_score,
    "loan_income_utilization_interaction": loan_income_utilization_interaction,
    "affordability_stress_interaction": affordability_stress_interaction
}])

st.subheader("Cooked Data: Engineered Features")
st.dataframe(input_df)

if st.button("Predict Default Risk"):
    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.metric("Default Probability", f"{probability:.2%}")
    st.metric("Predicted Class", "High Risk" if prediction == 1 else "Low Risk")

    if probability >= 0.70:
        st.error("This borrower appears high risk.")
    elif probability >= 0.40:
        st.warning("This borrower appears moderate risk.")
    else:
        st.success("This borrower appears lower risk.")