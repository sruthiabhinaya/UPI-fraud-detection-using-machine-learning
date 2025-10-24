import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # <- you need this
import matplotlib.pyplot as plt
import seaborn as sns

# Folder where your CSV is located
extract_to = "dataset"  # update if your CSV is in a different folder
csv_file = os.path.join(extract_to, "transactions.csv")

# Check if the file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found at {csv_file}")

# Load CSV
df = pd.read_csv(csv_file)

# Map Status column to numeric
df['Status'] = df['Status'].map({'FAILED': 0, 'SUCCESS': 1})

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

# Encode categorical columns
categorical_cols = ['Sender Name', 'Sender UPI ID', 'Receiver Name', 'Receiver UPI ID']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # ensure all values are string
    le_dict[col] = le

print("✅ Preprocessing done. Sample data:")
print(df.head())
