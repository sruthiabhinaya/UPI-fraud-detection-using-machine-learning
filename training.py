# isolation_forest_training.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("dataset/transactions.csv")
df.columns = df.columns.str.strip()

# Map target
df['Status'] = df['Status'].map({'FAILED': 0, 'SUCCESS': 1})

# 2️⃣ Create features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

# Aggregated sender/receiver features
df['Sender_txn_count'] = df.groupby('Sender UPI ID')['Amount (INR)'].transform('count')
df['Receiver_txn_count'] = df.groupby('Receiver UPI ID')['Amount (INR)'].transform('count')
df['Sender_failed_txn'] = df.groupby('Sender UPI ID')['Status'].transform('sum')
df['Receiver_failed_txn'] = df.groupby('Receiver UPI ID')['Status'].transform('sum')

# Clip counts to reduce extreme influence
df['Sender_txn_count'] = df['Sender_txn_count'].clip(0, 50)
df['Receiver_txn_count'] = df['Receiver_txn_count'].clip(0, 50)

# Features
features = ['Amount (INR)', 'Hour', 'DayOfWeek', 'Month',
            'Sender_txn_count', 'Receiver_txn_count',
            'Sender_failed_txn', 'Receiver_failed_txn']

X = df[features]

# 3️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Train IsolationForest
# contamination: fraction of outliers expected (adjust based on dataset)
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)

# 5️⃣ Save model and scaler
joblib.dump(model, "upi_fraud_iforest_model.pkl")
joblib.dump(scaler, "upi_iforest_scaler.pkl")

print("IsolationForest model trained and saved successfully.")
