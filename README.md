\# Credit Risk Prediction



\## Overview

This project predicts borrower default risk using financial, behavioural, and engineered credit risk features.



\## Business Problem

Lenders need to identify high-risk applicants before granting loans.



\## Features

The project includes:

\- loan\_to\_income\_ratio

\- debt\_to\_income\_ratio

\- credit\_utilization\_ratio

\- payment\_to\_income\_ratio

\- delinquency\_frequency

\- has\_delinquency\_flag

\- credit\_inquiry\_rate

\- employment\_stability\_flag

\- employment\_years\_bucket

\- financial\_stress\_score

\- behavioral\_risk\_score

\- loan\_income\_utilization\_interaction

\- affordability\_stress\_interaction



\## Project Structure

\- `src/` model code

\- `app/` Streamlit dashboard

\- `data/raw/` input data

\- `outputs/models/` saved model

\- `outputs/figures/` charts



\## How to Run



\### Train

```bash

python src/train.py

