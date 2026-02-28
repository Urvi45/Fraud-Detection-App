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
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

# -----------------------------
# Load Files
# -----------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Safe Encoding
# -----------------------------
def safe_label_encode(column, encoder):
    known_classes = list(encoder.classes_)
    return column.apply(
        lambda x: encoder.transform([x])[0] if x in known_classes else -1
    )

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    # Encode merchant_category
    if "merchant_category" in df.columns:
        df["merchant_category"] = safe_label_encode(df["merchant_category"], le)

    # Validate columns
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    df_features = df[feature_columns]

    # Scale
    df_scaled = scaler.transform(df_features)

    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    df["Prediction"] = predictions
    df["Fraud_Probability"] = probabilities

    # -----------------------------
    # SHOW PREDICTION TABLE FIRST
    # -----------------------------
    st.subheader("🔍 Prediction Results")
    st.dataframe(df.head(10), use_container_width=True)

    # -----------------------------
    # Metrics
    # -----------------------------
    fraud_count = sum(predictions)
    total = len(predictions)
    fraud_percentage = (fraud_count / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Fraud", fraud_count)
    col3.metric("Fraud %", f"{fraud_percentage:.2f}%")

    st.markdown("---")

    # -----------------------------
    # Small Graphs (Compact Size)
    # -----------------------------
    col4, col5 = st.columns(2)

    # Pie Chart
    with col4:
        st.subheader("Fraud Distribution")
        plt.figure(figsize=(4,3))
        plt.pie(
            [total - fraud_count, fraud_count],
            labels=["Normal", "Fraud"],
            autopct='%1.1f%%',
            colors=["#4CAF50", "#F44336"]
        )
        plt.axis("equal")
        st.pyplot(plt)

    # Probability Histogram
    with col5:
        st.subheader("Fraud Probability")
        plt.figure(figsize=(4,3))
        sns.histplot(df["Fraud_Probability"], bins=20, kde=True)
        st.pyplot(plt)

    # -----------------------------
    # ROC Curve (Small)
    # -----------------------------
    if "is_fraud" in df.columns:
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(df["is_fraud"], probabilities)
        roc_score = roc_auc_score(df["is_fraud"], probabilities)

        plt.figure(figsize=(4,3))
        plt.plot(fpr, tpr, label=f"AUC = {roc_score:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.legend()
        st.pyplot(plt)

    # -----------------------------
    # Download Button
    # -----------------------------
    st.download_button(
        "📥 Download Results",
        data=df.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to start fraud detection.")