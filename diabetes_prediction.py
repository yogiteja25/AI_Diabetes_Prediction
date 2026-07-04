# ==========================================================
# AI-Based Diabetes Prediction System
# diabetes_prediction.py
# PART 1
# ==========================================================

# ===============================
# Import Libraries
# ===============================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===============================
# Create Required Folders
# ===============================

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ===============================
# Load Dataset
# ===============================

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv("dataset/diabetes_binary_health.csv")

print("\nDataset Loaded Successfully\n")

# ===============================
# Display Dataset
# ===============================

print("First Five Records\n")

print(df.head())

print("\n")

print("Last Five Records\n")

print(df.tail())

# ===============================
# Dataset Shape
# ===============================

print("\nDataset Shape")

print(df.shape)

# ===============================
# Dataset Information
# ===============================

print("\nDataset Information")

df.info()

# ===============================
# Column Names
# ===============================

print("\nColumns\n")

print(df.columns)

# ===============================
# Statistical Summary
# ===============================

print("\nStatistical Summary\n")

print(df.describe())

# ===============================
# Missing Values
# ===============================

print("\nMissing Values\n")

print(df.isnull().sum())

# Fill Missing Values

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing Values After Filling\n")

print(df.isnull().sum())

# ===============================
# Duplicate Records
# ===============================

print("\nDuplicate Records")

print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates")

print(df.shape)

# ==========================================================
# Data Visualization
# ==========================================================

print("\nGenerating Graphs...\n")

# -----------------------------
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("images/heatmap.png")

plt.show()

# -----------------------------
# Diabetes Distribution
# -----------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="Diabetes",
    data=df
)

plt.title("Diabetes Distribution")

plt.savefig("images/diabetes_distribution.png")

plt.show()

# -----------------------------
# BMI Distribution
# -----------------------------

plt.figure(figsize=(6,4))

sns.histplot(
    df["BMI"],
    bins=20,
    kde=True
)

plt.title("BMI Distribution")

plt.savefig("images/bmi_distribution.png")

plt.show()

# -----------------------------
# Age Distribution
# -----------------------------

plt.figure(figsize=(6,4))

sns.histplot(
    df["Age"],
    bins=20,
    kde=True
)

plt.title("Age Distribution")

plt.savefig("images/age_distribution.png")

plt.show()

# -----------------------------
# Glucose Distribution
# -----------------------------

plt.figure(figsize=(6,4))

sns.histplot(
    df["Glucose"],
    bins=20,
    kde=True
)

plt.title("Glucose Distribution")

plt.savefig("images/glucose_distribution.png")

plt.show()

# -----------------------------
# Blood Pressure Distribution
# -----------------------------

plt.figure(figsize=(6,4))

sns.histplot(
    df["BloodPressure"],
    bins=20,
    kde=True
)

plt.title("Blood Pressure Distribution")

plt.savefig("images/blood_pressure_distribution.png")

plt.show()

# ==========================================================
# Feature Selection
# ==========================================================

print("\nPreparing Features and Target\n")

X = df[[
    "Age",
    "BMI",
    "PhysicalActivity",
    "BloodPressure",
    "Cholesterol",
    "Glucose"
]]

y = df["Diabetes"]

print("Feature Shape :", X.shape)

print("Target Shape :", y.shape)

# ==========================================================
# Train Test Split
# ==========================================================

print("\nSplitting Dataset\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data :", X_train.shape)

print("Testing Data :", X_test.shape)

# ==========================================================
# Feature Scaling
# ==========================================================

print("\nApplying Standard Scaler\n")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Scaling Completed Successfully")

print("\nPART 1 COMPLETED SUCCESSFULLY")
# ==========================================================
# PART 2
# Model Training, Evaluation and Saving
# ==========================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import joblib

print("="*60)
print("TRAINING RANDOM FOREST MODEL")
print("="*60)

# ==========================================================
# Create Model
# ==========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================================
# Train Model
# ==========================================================

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully")

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

print("\nPrediction Completed Successfully")

# ==========================================================
# Evaluation Metrics
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob[:,1])

print("\n")
print("="*60)
print("MODEL EVALUATION")
print("="*60)

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print(f"ROC-AUC   : {roc_auc*100:.2f}%")

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("images/confusion_matrix.png")

plt.show()

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, threshold = roc_curve(
    y_test,
    y_prob[:,1]
)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="Random Forest")

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig("images/roc_curve.png")

plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

importance = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature":feature_names,
    "Importance":importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")

print(importance_df)

plt.figure(figsize=(8,5))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    model,
    "models/diabetes_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully")

print("Location : models/diabetes_model.pkl")

print("Scaler Saved Successfully")

# ==========================================================
# Sample Prediction
# ==========================================================

print("\n")
print("="*60)
print("SAMPLE PREDICTION")
print("="*60)

sample = pd.DataFrame({
    "Age":[45],
    "BMI":[31],
    "PhysicalActivity":[2],
    "BloodPressure":[145],
    "Cholesterol":[220],
    "Glucose":[180]
})

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

probability = model.predict_proba(sample_scaled)

print("\nPatient Details")

print(sample)

print("\nPrediction")

if prediction[0] == 1:
    print("Diabetic")
else:
    print("Non-Diabetic")

print(f"Risk Percentage : {probability[0][1]*100:.2f}%")

print("="*60)
print("PROJECT EXECUTED SUCCESSFULLY")
print("="*60)