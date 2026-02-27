import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection Dashboard")

# ---------------------------
# Load Model & Scaler
# ---------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Exact training feature order
feature_columns = [
    "id",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount"
]

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Validate columns
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    # Keep correct column order
    df_features = df[feature_columns]

    # Scale
    df_scaled = scaler.transform(df_features)

    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    df["Prediction"] = predictions
    df["Fraud_Probability"] = probabilities

    # ---------------------------
    # Metrics Section
    # ---------------------------
    fraud_count = sum(predictions)
    total = len(predictions)
    fraud_percentage = (fraud_count / total) * 100

    st.markdown("---")
    st.subheader("📊 Fraud Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Transactions", fraud_count)
    col3.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

    st.markdown("---")

    # ---------------------------
    # Charts Section
    # ---------------------------
    col4, col5 = st.columns(2)

    # Pie Chart
    with col4:
        st.subheader("Fraud Distribution")
        fig1, ax1 = plt.subplots(figsize=(4,4))
        ax1.pie(
            [total - fraud_count, fraud_count],
            labels=["Normal", "Fraud"],
            autopct='%1.1f%%',
            colors=["#4CAF50", "#F44336"],
            startangle=90
        )
        ax1.axis("equal")
        st.pyplot(fig1)

    # ROC Curve (if Class available)
    if "Class" in df.columns:
        with col5:
            st.subheader("ROC Curve")
            y_true = df["Class"]

            fpr, tpr, _ = roc_curve(y_true, probabilities)
            roc_score = roc_auc_score(y_true, probabilities)

            fig2, ax2 = plt.subplots(figsize=(4,4))
            ax2.plot(fpr, tpr, color="#1f77b4", linewidth=2)
            ax2.plot([0,1], [0,1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title(f"AUC = {roc_score:.4f}")
            st.pyplot(fig2)

    st.markdown("---")

    # ---------------------------
    # Download Section
    # ---------------------------
    st.subheader("📥 Download Results")

    st.download_button(
        label="Download Prediction CSV",
        data=df.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to start fraud detection.")