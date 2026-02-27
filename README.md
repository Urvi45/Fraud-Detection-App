# 💳 Credit Card Fraud Detection System

🔗 **Live Deployment Link:**  
https://urvi45-fraud-detection-app-app-usghvv.streamlit.app/

---

## 📌 Project Overview

This project is a **Machine Learning-based Credit Card Fraud Detection System** developed as a final-year project.

The system analyzes transaction data and predicts whether a transaction is:

- ✅ Normal (0)
- ⚠ Fraudulent (1)

The model is trained using the **Gradient Boosting Classifier** and deployed using **Streamlit** for real-time batch transaction analysis.

---

## 🚀 Live Application

You can access the deployed application here:

👉 https://urvi45-fraud-detection-app-app-usghvv.streamlit.app/

The app allows:
- CSV file upload
- Batch transaction prediction
- Fraud probability calculation
- Fraud percentage visualization
- ROC Curve (if Class column exists)
- Downloadable prediction results

---

## 📁 Project Structure

```bash
│
├── app.py # Main Streamlit application
├── fraud_model.pkl # Trained ML model
├── scaler.pkl # StandardScaler used during training
├── requirements.txt # Python dependencies
├── feature_columns.pkl # Feature list used for prediction
├── sample_transactions.csv # Sample CSV for testing
└── README.md # Project documentation
```

---

## 📊 About the Dataset

The dataset contains credit card transaction records with the following features:

| Column | Description |
|--------|-------------|
| id     | Unique transaction identifier |
| V1–V28 | PCA-transformed anonymized features |
| Amount | Transaction amount |
| Class  | Target variable (0 = Normal, 1 = Fraud) |

### 🔎 Important Notes:
- Features V1–V28 are transformed using **Principal Component Analysis (PCA)** to protect sensitive information.
- The dataset is **highly imbalanced**, meaning fraud cases are very rare compared to normal transactions.
- The `Class` column is used only during training and evaluation.

---

## 🧠 What Was Done in the Jupyter Notebook (Model Development)

The following steps were performed during model development:

### 1️⃣ Data Exploration
- Checked dataset structure
- Analyzed class imbalance
- Reviewed feature distributions

### 2️⃣ Data Preprocessing
- Removed unnecessary columns if needed
- Feature scaling using `StandardScaler`
- Train-Test split

### 3️⃣ Model Training
- Used `GradientBoostingClassifier`
- Manually tuned hyperparameters
- Handled imbalanced data carefully

### 4️⃣ Model Evaluation
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Score
- ROC Curve visualization

### 5️⃣ Model Saving
- Saved trained model using `joblib`
- Saved scaler
- Saved feature column order for deployment

---

## 📥 How to Use the Application

1. Prepare a CSV file with the following columns in exact order:
```bash
id, V1, V2, ..., V28, Amount
```

2. Upload the CSV file in the Streamlit app.
3. The system will:
   - Predict fraud status
   - Calculate fraud probability
   - Display fraud percentage
   - Show charts
   - Allow downloading predictions

If your CSV includes the `Class` column, the system will also display the ROC curve.

---
