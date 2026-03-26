import numpy as np
import pandas as pd


def safe_divide(a, b):
    return a / b.replace(0, np.nan)


def create_features(df):
    df = df.copy()

    # Basic ratios
    df["loan_to_income_ratio"] = safe_divide(df["loan_amount"], df["annual_income"])
    df["debt_to_income_ratio"] = safe_divide(df["total_debt"], df["annual_income"])
    df["credit_utilization_ratio"] = safe_divide(df["credit_balance"], df["credit_limit"])

    # Monthly affordability
    monthly_income = df["annual_income"] / 12
    df["payment_to_income_ratio"] = safe_divide(df["monthly_payment"], monthly_income)

    # Behaviour
    df["delinquency_frequency"] = safe_divide(df["num_delinquencies"], df["credit_history_length"])
    df["has_delinquency_flag"] = (df["num_delinquencies"] > 0).astype(int)

    # Credit seeking
    df["credit_inquiry_rate"] = safe_divide(df["num_credit_inquiries"], df["credit_history_length"])

    # Employment stability
    df["employment_stability_flag"] = (df["employment_years"] >= 3).astype(int)

    df["employment_years_bucket"] = pd.cut(
        df["employment_years"],
        bins=[-np.inf, 1, 3, 5, 10, np.inf],
        labels=["0-1", "1-3", "3-5", "5-10", "10+"]
    )

    # Composite scores
    df["financial_stress_score"] = (
        0.5 * df["debt_to_income_ratio"].fillna(0) +
        0.5 * df["credit_utilization_ratio"].fillna(0)
    )

    df["behavioral_risk_score"] = (
        0.4 * df["debt_to_income_ratio"].fillna(0) +
        0.3 * df["credit_utilization_ratio"].fillna(0) +
        0.3 * df["delinquency_frequency"].fillna(0)
    )

    # Interaction features
    df["loan_income_utilization_interaction"] = (
        df["loan_to_income_ratio"].fillna(0) *
        df["credit_utilization_ratio"].fillna(0)
    )

    df["affordability_stress_interaction"] = (
        df["payment_to_income_ratio"].fillna(0) *
        df["debt_to_income_ratio"].fillna(0)
    )

    # Demo target creation
    df["default_flag"] = (df["behavioral_risk_score"] > 0.40).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df