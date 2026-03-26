import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("📊 Credit Risk Prediction Dashboard")

st.markdown("### Business Overview")
st.write("This dashboard predicts the likelihood of customer default.")

# Sample data (replace later with your real model)
data = pd.DataFrame({
    "Age": np.random.randint(21, 65, 100),
    "Income": np.random.randint(2000, 20000, 100),
    "Loan Amount": np.random.randint(1000, 15000, 100),
    "Default": np.random.choice([0, 1], 100)
})

st.markdown("### Dataset Preview")
st.dataframe(data.head())

st.markdown("### Default Distribution")
st.bar_chart(data["Default"].value_counts())

st.markdown("### Feature Relationships")
st.scatter_chart(data, x="Income", y="Loan Amount")

st.success("Dashboard running successfully 🚀")