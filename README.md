# Credit Card Fraud Analytics Platform

A machine learning-driven analytics platform designed to identify potentially fraudulent credit card transactions and provide actionable insights through risk scoring, performance evaluation, and interactive visualizations.

### Live Demo

[https://credit-card-fraud-analytics-ccfa.streamlit.app/]

### Repository

[https://github.com/Dkiranmayee25/credit-card-fraud-analytics]

---

## Overview

Credit card fraud continues to be a major challenge for financial institutions due to the high volume of daily transactions and the evolving nature of fraudulent behavior. This project presents an end-to-end fraud analytics solution that leverages machine learning to classify transactions, estimate fraud risk, and support data-driven investigation workflows.

The platform enables users to upload transaction datasets, analyze fraud patterns, evaluate model performance, and export prediction reports through an intuitive web-based interface.

---

## Features

* Batch transaction analysis through CSV uploads
* Fraud prediction using a trained XGBoost model
* Fraud risk scoring and risk-level categorization
* Interactive analytics dashboard
* Fraud distribution visualization
* Risk score distribution analysis
* High-risk transaction identification
* Downloadable prediction reports
* Model performance evaluation using industry-standard metrics

---

## Dashboard Capabilities

### Transaction Analysis

* Upload transaction datasets for automated fraud assessment
* Generate fraud predictions at scale
* View transaction-level risk scores

### Risk Intelligence

* Categorize transactions into:

  * Low Risk
  * Medium Risk
  * High Risk

### Performance Monitoring

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Visualization

* Fraud vs Legitimate Transaction Distribution
* Risk Score Distribution
* Fraud Investigation Tables
* Interactive Performance Insights

---

## System Workflow

```text
Transaction Dataset
        │
        ▼
Data Validation
        │
        ▼
Feature Processing
        │
        ▼
XGBoost Classification Model
        │
        ▼
Fraud Prediction
        │
        ▼
Risk Scoring
        │
        ▼
Analytics Dashboard
        │
        ▼
Report Generation
```

## Dataset

The model was trained on an anonymized credit card transaction dataset containing legitimate and fraudulent transactions.

To preserve confidentiality, most transaction attributes were transformed using Principal Component Analysis (PCA) and are represented as:

```text
V1, V2, V3, ..., V28
```

Additional available features include:

* Time
* Amount

Target Variable:

```text
Class
0 → Legitimate Transaction
1 → Fraudulent Transaction
```

### Dataset Characteristics

* Binary Classification Problem
* Highly Imbalanced Dataset
* 30 Input Features
* Real-world Fraud Detection Scenario

---

## Technology Stack

| Category             | Technologies          |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Machine Learning     | XGBoost, Scikit-learn |
| Data Processing      | Pandas, NumPy         |
| Visualization        | Plotly                |
| Web Application      | Streamlit             |
| Model Serialization  | Joblib                |

---

## Project Structure

```text
credit-card-fraud-analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    └── xgboost.pkl
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-analytics.git
cd credit-card-fraud-analytics
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Expected Input Format

The uploaded transaction file should contain the following columns:

```text
Time, V1, V2, V3, ..., V28, Amount
```

Optional:

```text
Class
```

If the Class column is provided, the platform automatically computes model evaluation metrics such as Accuracy, Precision, Recall, and F1 Score.

---

## Model Evaluation

The platform supports multiple evaluation metrics for assessing classification performance:

| Metric           | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Accuracy         | Overall prediction correctness            |
| Precision        | Reliability of fraud predictions          |
| Recall           | Ability to detect fraudulent transactions |
| F1 Score         | Balance between Precision and Recall      |
| Confusion Matrix | Detailed classification analysis          |

These metrics are particularly important for fraud detection due to the highly imbalanced nature of financial transaction data.

---

## Use Cases

* Fraud Analytics
* Financial Risk Assessment
* Transaction Monitoring
* Model Performance Evaluation
* Machine Learning Demonstration
* Educational and Research Applications

---

## Future Enhancements

* Real-time transaction monitoring
* Explainable AI integration using SHAP
* Database-backed transaction history
* Automated fraud alerting
* Advanced risk profiling
* Cloud-native deployment architecture
* Role-based access and authentication

---

## Disclaimer

This project is intended for educational, research, and demonstration purposes. The dataset used contains anonymized transaction features and does not expose any personally identifiable customer information.

---

## Author

Kiranmayee D

Computer Science Undergraduate

Interests: Machine Learning, Data Analytics, Artificial Intelligence, and Software Development
