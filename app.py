import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Credit Card Fraud Detection System")

# -----------------------------
# Load Saved Files
# -----------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Safe Encoding
# -----------------------------
def safe_label_encode(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1

# -----------------------------
# Sidebar Mode Selection
# -----------------------------
mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    ["Single Transaction Analysis", "Bulk CSV Analysis"]
)

# =====================================================
# 🔵 1️⃣ SINGLE TRANSACTION ANALYSIS
# =====================================================
if mode == "Single Transaction Analysis":

    st.subheader("🔍 Analyze Single Transaction")

    transaction_id = st.number_input("Transaction ID", value=1)
    amount = st.number_input("Amount", value=100.0)
    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
    merchant_category = st.selectbox(
        "Merchant Category",
        list(le.classes_)
    )
    foreign_transaction = st.selectbox("Foreign Transaction", [0, 1])
    location_mismatch = st.selectbox("Location Mismatch", [0, 1])
    device_trust_score = st.slider("Device Trust Score", 0, 100, 50)
    velocity_last_24h = st.slider("Transactions in Last 24h", 0, 50, 5)
    cardholder_age = st.slider("Cardholder Age", 18, 80, 30)

    if st.button("Predict Fraud"):

        encoded_category = safe_label_encode(merchant_category, le)

        input_data = pd.DataFrame([[
            amount,
            transaction_hour,
            encoded_category,
            foreign_transaction,
            location_mismatch,
            device_trust_score,
            velocity_last_24h,
            cardholder_age
        ]], columns=[
            "amount",
            "transaction_hour",
            "merchant_category",
            "foreign_transaction",
            "location_mismatch",
            "device_trust_score",
            "velocity_last_24h",
            "cardholder_age"
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error(f"⚠ Fraud Detected! (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Normal Transaction (Probability: {probability:.2f})")

# =====================================================
# 🟢 2️⃣ BULK CSV ANALYSIS
# =====================================================
else:

    st.subheader("📂 Bulk Transaction Analysis")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # Encode merchant category safely
        if "merchant_category" in df.columns:
            df["merchant_category"] = df["merchant_category"].apply(
                lambda x: safe_label_encode(x, le)
            )

        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        df_features = df[feature_columns]
        df_scaled = scaler.transform(df_features)

        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        df["Prediction"] = predictions
        df["Fraud_Probability"] = probabilities

        # Show table first
        st.subheader("🔍 Prediction Results")
        st.dataframe(df.head(10), use_container_width=True)

        # Metrics
        fraud_count = sum(predictions)
        total = len(predictions)
        fraud_percentage = (fraud_count / total) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Fraud", fraud_count)
        col3.metric("Fraud %", f"{fraud_percentage:.2f}%")

        st.markdown("---")

        # Small Charts
        col4, col5 = st.columns(2)

        with col4:
            st.subheader("Fraud Distribution")
            plt.figure(figsize=(4,3))
            plt.pie(
                [total - fraud_count, fraud_count],
                labels=["Normal", "Fraud"],
                autopct='%1.1f%%'
            )
            plt.axis("equal")
            st.pyplot(plt)

        with col5:
            st.subheader("Fraud Probability")
            plt.figure(figsize=(4,3))
            sns.histplot(df["Fraud_Probability"], bins=20, kde=True)
            st.pyplot(plt)

        # Download
        st.download_button(
            "📥 Download Results",
            data=df.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )