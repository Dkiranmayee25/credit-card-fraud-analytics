
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import plotly.figure_factory as ff
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Analytics",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
try:
    model = joblib.load("models/xgboost.pkl")
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

# --------------------------------------------------
# CUSTOM STYLE
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
        color:#1f77b4;
    }
    .sub-title {
        text-align:center;
        color:gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Page",
    ["Dashboard", "Batch Analysis", "About Project"]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    st.markdown(
        "<div class='main-title'>💳 Credit Card Fraud Analytics Platform</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='sub-title'>Machine Learning Based Fraud Detection System</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "XGBoost")
    col2.metric("Features", "30")
    col3.metric("Prediction Type", "Fraud Detection")

    st.markdown("---")
    st.subheader("Project Overview")
    st.write(
        """
        This platform detects fraudulent credit card transactions
        using a trained XGBoost machine learning model.

        Features:
        • Batch transaction analysis
        • Fraud risk scoring
        • Fraud statistics
        • Data visualization
        • Downloadable prediction reports
        """
    )

# --------------------------------------------------
# BATCH ANALYSIS
# --------------------------------------------------
elif page == "Batch Analysis":

    st.title("📁 Batch Fraud Analysis")

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file:

        try:

            # Read file once
            original_df = pd.read_csv(uploaded_file)

            # Keep labels if available
            has_labels = "Class" in original_df.columns

            # Create prediction dataframe
            df = original_df.copy()

            if "Class" in df.columns:
                df = df.drop(columns=["Class"])

            # Expected features
            expected_columns = [
                "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
                "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
                "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
            ]

            if list(df.columns) != expected_columns:
                st.error(
                    "Uploaded file must contain: Time, V1-V28, Amount"
                )
                st.stop()

            st.subheader("Uploaded Data")
            st.dataframe(df.head())

            # Predictions
            predictions = model.predict(df)

            try:
                probabilities = model.predict_proba(df)[:, 1]
            except Exception:
                probabilities = np.zeros(len(df))

            df["Prediction"] = predictions
            df["Risk Score"] = (probabilities * 100).round(2)

            # Risk Levels
            def risk_level(score):
                if score >= 80:
                    return "HIGH"
                elif score >= 40:
                    return "MEDIUM"
                return "LOW"

            df["Risk Level"] = df["Risk Score"].apply(risk_level)

            # Model Metrics (only if Class exists)
            if has_labels:

                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    confusion_matrix
                )

                accuracy = accuracy_score(
                    original_df["Class"],
                    predictions
                )

                precision = precision_score(
                    original_df["Class"],
                    predictions
                )

                recall = recall_score(
                    original_df["Class"],
                    predictions
                )

                f1 = f1_score(
                    original_df["Class"],
                    predictions
                )

                st.subheader("Model Performance")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Accuracy", f"{accuracy:.3f}")
                c2.metric("Precision", f"{precision:.3f}")
                c3.metric("Recall", f"{recall:.3f}")
                c4.metric("F1 Score", f"{f1:.3f}")

                # Confusion Matrix
                cm = confusion_matrix(
                    original_df["Class"],
                    predictions
                )

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    title="Confusion Matrix",
                    labels=dict(
                        x="Predicted",
                        y="Actual"
                    )
                )

                st.plotly_chart(
                    fig_cm,
                    use_container_width=True
                )

            # Statistics
            fraud_count = int(df["Prediction"].sum())
            genuine_count = len(df) - fraud_count
            fraud_rate = (fraud_count / len(df)) * 100

            st.subheader("Fraud Statistics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fraud Transactions",
                fraud_count
            )

            col2.metric(
                "Legitimate Transactions",
                genuine_count
            )

            col3.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%"
            )

            # Pie Chart
            fig1 = px.pie(
                names=["Legitimate", "Fraud"],
                values=[genuine_count, fraud_count],
                title="Fraud Distribution"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            # Risk Distribution
            fig2 = px.histogram(
                df,
                x="Risk Score",
                nbins=20,
                title="Risk Score Distribution"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            # High Risk Transactions
            st.subheader(
                "⚠ Predicted Fraud Transactions"
            )

            fraud_df = df[
                df["Prediction"] == 1
            ]

            if len(fraud_df) > 0:
                st.dataframe(fraud_df)
            else:
                st.success(
                    "No fraudulent transactions detected."
                )

            # Full Results
            st.subheader("All Results")
            st.dataframe(df)

            # Download
            csv = df.to_csv(index=False)

            st.download_button(
                "⬇ Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif page == "About Project":
    st.title("ℹ About")
    st.write(
        """
        Credit Card Fraud Analytics Platform

        Technologies Used:
        - Python
        - Streamlit
        - XGBoost
        - Pandas
        - NumPy
        - Plotly
        - Joblib

        Purpose:
        Detect fraudulent credit card transactions
        and provide fraud analytics insights.
        """
    )
