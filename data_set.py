# upi_dataset_exploration_only.py

import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# STEP 1: Load & Extract Dataset
# -----------------------------
zip_path = "archive.zip"   # Change to your ZIP file name
extract_to = "dataset"

# Extract ZIP if folder doesn't exist
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print("✅ Extracted ZIP to:", extract_to)

# List files
print("Extracted files:", os.listdir(extract_to))

# Load CSV
csv_file = os.path.join(extract_to, "transactions.csv")
df = pd.read_csv(csv_file)
print("\n📂 Dataset Loaded — Shape:", df.shape)

# -----------------------------
# STEP 2: Explore the Data
# -----------------------------
print("\nℹ️ Data Info:")
print(df.info())

print("\n🔢 Descriptive Statistics:")
print(df.describe())

# Check missing values
print("\n❗ Missing Values:")
print(df.isnull().sum())

# Check class distribution (Status column)
print("\nClass Distribution (Status):")
print(df['Status'].value_counts())
print(df['Status'].value_counts(normalize=True))

# Visualize class imbalance
plt.figure(figsize=(6,4))
sns.countplot(x='Status', data=df)
plt.title("Transaction Status Distribution")
plt.show()

# Numeric feature distributions
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(figsize=(12,8), bins=20)
plt.suptitle("Numeric Feature Distributions")
plt.show()

# Correlation heatmap (numeric columns only)
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Example: Top Senders by number of transactions
if 'Sender UPI ID' in df.columns:
    top_senders = df['Sender UPI ID'].value_counts().head(10)
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_senders.index, y=top_senders.values)
    plt.title("Top 10 Senders by Number of Transactions")
    plt.xlabel("Sender UPI ID")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=45)
    plt.show()
