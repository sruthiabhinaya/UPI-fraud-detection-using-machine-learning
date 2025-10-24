# gradio_upi_fraud.py

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("upi_fraud_iforest_model.pkl")
scaler = joblib.load("upi_iforest_scaler.pkl")
historical_df = pd.read_csv("dataset/transactions.csv")
historical_df.columns = historical_df.columns.str.strip()

def preprocess_transaction(txn):
    df = pd.DataFrame([txn])
    
    # Timestamp features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    
    # Aggregated sender/receiver counts
    sender_id = txn['Sender UPI ID']
    receiver_id = txn['Receiver UPI ID']
    
    sender_txns = historical_df[historical_df['Sender UPI ID'] == sender_id]
    receiver_txns = historical_df[historical_df['Receiver UPI ID'] == receiver_id]
    
    df['Sender_txn_count'] = len(sender_txns) if len(sender_txns) > 0 else 1
    df['Receiver_txn_count'] = len(receiver_txns) if len(receiver_txns) > 0 else 1
    df['Sender_failed_txn'] = sender_txns['Status'].sum() if len(sender_txns) > 0 else 0
    df['Receiver_failed_txn'] = receiver_txns['Status'].sum() if len(receiver_txns) > 0 else 0
    
    # Features only
    df = df[['Amount (INR)', 'Hour', 'DayOfWeek', 'Month',
             'Sender_txn_count', 'Receiver_txn_count',
             'Sender_failed_txn', 'Receiver_failed_txn']]
    
    X_scaled = scaler.transform(df)
    return X_scaled

def predict_fraud(Transaction_ID, Timestamp, Sender_Name, Sender_UPI, 
                  Receiver_Name, Receiver_UPI, Amount_INR):
    
    txn = {
        'Transaction ID': Transaction_ID,
        'Timestamp': Timestamp,
        'Sender Name': Sender_Name,
        'Sender UPI ID': Sender_UPI,
        'Receiver Name': Receiver_Name,
        'Receiver UPI ID': Receiver_UPI,
        'Amount (INR)': float(Amount_INR)
    }
    
    X_scaled = preprocess_transaction(txn)
    pred = model.predict(X_scaled)[0]  # 1 = normal, -1 = anomaly

    if pred == 1:
        return "NOT FRAUD"
    else:
        return "FRAUD"

# Gradio Interface
inputs = [
    gr.Textbox(label="Transaction ID"),
    gr.Textbox(label="Timestamp (YYYY-MM-DD HH:MM:SS)"),
    gr.Textbox(label="Sender Name"),
    gr.Textbox(label="Sender UPI ID"),
    gr.Textbox(label="Receiver Name"),
    gr.Textbox(label="Receiver UPI ID"),
    gr.Number(label="Amount (INR)")
]

outputs = gr.Label(label="Fraud Prediction")

gr.Interface(
    fn=predict_fraud,
    inputs=inputs,
    outputs=outputs,
    title="UPI Fraud Detection",
    description="Detects if a UPI transaction is FRAUD or NOT FRAUD.",
).launch()
