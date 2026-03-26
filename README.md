# 💳 Credit Risk Prediction & Explainability Dashboard

## 🚀 Overview
This project builds an end-to-end **credit risk prediction system** that estimates the probability of borrower default using financial, behavioural, and engineered features.

It includes:
- Feature engineering (domain-driven)
- Machine learning model (Random Forest)
- Model evaluation (ROC-AUC, classification metrics)
- Explainability using SHAP
- Interactive Streamlit dashboard

---

## 🧠 Business Problem
Financial institutions need to assess borrower risk before approving loans.

Poor decisions lead to:
- Increased default rates
- Financial losses
- Poor capital allocation

This project helps:
👉 Identify high-risk borrowers  
👉 Improve approval decisions  
👉 Support risk-based pricing  

---

## ⚙️ Key Features Engineered

### 💸 Affordability
- Debt-to-Income Ratio
- Loan-to-Income Ratio
- Payment-to-Income Ratio

### 💳 Utilization
- Credit Utilization Ratio

### ⚠️ Behaviour
- Delinquency Frequency
- Delinquency Flag

### 🔍 Credit Activity
- Inquiry Rate

### 🧑‍💼 Stability
- Employment Stability
- Employment Buckets

### 🧠 Composite Risk Scores
- Financial Stress Score
- Behavioral Risk Score

### 🔗 Interaction Features
- Loan × Utilization
- Affordability × Debt Stress

---

## 📊 Model Performance

- Model: Random Forest Classifier
- Metric: ROC-AUC
- Handles class imbalance using `class_weight="balanced"`

---

## 🔍 Explainability (SHAP)

The project uses SHAP to:
- Identify most important features
- Explain model predictions
- Improve transparency for stakeholders

---

## 🖥️ Dashboard

Interactive Streamlit dashboard allows users to:
- Input borrower data
- View engineered features
- Predict default probability
- Interpret results in business terms

## 📸 Dashboard Preview

Below is a preview of the interactive credit risk dashboard:

![Dashboard Preview](images/dashboard-preview.png)

---

## 📸 Dashboard Preview

![Dashboard](outputs/figures/dashboard.png)

---

## 🏗️ Project Structure
