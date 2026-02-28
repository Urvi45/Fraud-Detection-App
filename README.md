# 💳 Credit Card Fraud Detection System

🔗 **Live Application:**  
👉 https://urvi45-fraud-detection-app-app-usghvv.streamlit.app/

---

## 📌 Project Overview

This project is a **Machine Learning-based Credit Card Fraud Detection System** developed as a final-year project.

The system analyzes transaction data and predicts whether a transaction is:

- ✅ Normal (0)
- ⚠ Fraudulent (1)

The model is trained using **Gradient Boosting Classifier** and deployed using **Streamlit** to provide an interactive fraud detection dashboard.

---

## 🚀 Features

✔ Upload transaction CSV file  
✔ Batch fraud prediction  
✔ Fraud probability score  
✔ Fraud percentage calculation  
✔ Fraud distribution pie chart  
✔ Fraud probability histogram  
✔ ROC curve (if actual labels are available)  
✔ Downloadable prediction results  

---
```bush
│
├── app.py # Main Streamlit dashboard
├── fraud_model.pkl # Trained Gradient Boosting model
├── scaler.pkl # StandardScaler used during training
├── feature_columns.pkl # Feature order used for prediction
├── label_encoder.pkl # Encoder for merchant_category
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## 📊 Dataset Description

The dataset contains 10,000 transaction records with the following features:

| Column | Description |
|--------|------------|
| transaction_id | Unique transaction identifier |
| amount | Transaction amount |
| transaction_hour | Hour of transaction (0–23) |
| merchant_category | Merchant type/category |
| foreign_transaction | Foreign transaction flag (0/1) |
| location_mismatch | Location mismatch flag (0/1) |
| device_trust_score | Device reliability score |
| velocity_last_24h | Transactions in last 24 hours |
| cardholder_age | Age of cardholder |
| is_fraud | Target variable (0 = Normal, 1 = Fraud) |

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Preprocessing
- Removed unnecessary column (`transaction_id`)
- Encoded categorical column (`merchant_category`)
- Applied feature scaling using `StandardScaler`
- Split dataset into train and test sets

### 2️⃣ Model Training
- Algorithm: **Gradient Boosting Classifier**
- Parameters:
  - n_estimators = 100
  - learning_rate = 0.1
  - max_depth = 3
- Handled class imbalance using proper evaluation metrics

### 3️⃣ Model Evaluation
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Score
- ROC Curve

### 4️⃣ Model Deployment
- Saved model using `joblib`
- Built interactive dashboard using Streamlit
- Deployed via GitHub + Streamlit Community Cloud

---

## 📈 Dashboard Visualizations

The deployed app includes:

- 📊 Fraud vs Normal Pie Chart
- 📈 Fraud Probability Histogram
- 📉 ROC Curve (if label provided)
- 📊 Fraud metrics summary
- 📥 Downloadable prediction file
