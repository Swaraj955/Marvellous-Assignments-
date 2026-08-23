# ============================================================
# BREAST CANCER PREDICTION
# Breast Cancer Wisconsin Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Copy of breast-cancer-wisconsin.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ============================================================
# 2. EXPLORE DATASET
# ============================================================

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

print("\nMissing Values Before Processing:")
print(df.isnull().sum())

# Replace '?' with NaN
df["BareNuclei"] = df["BareNuclei"].replace("?", np.nan)

# Convert BareNuclei into numeric
df["BareNuclei"] = pd.to_numeric(df["BareNuclei"])

print("\nMissing Values After Replacing '?':")
print(df.isnull().sum())

# Fill missing values with median
df["BareNuclei"] = df["BareNuclei"].fillna(df["BareNuclei"].median())

print("\nMissing Values After Processing:")
print(df.isnull().sum())

# ============================================================
# 4. TARGET VARIABLE
# ============================================================

print("\nOriginal Target Values:")
print(df["CancerType"].value_counts())

# 2 = Benign -> 1
# 4 = Malignant -> 0
df["CancerType"] = df["CancerType"].map({
    2: 1,
    4: 0
})

print("\nTarget Values After Conversion:")
print(df["CancerType"].value_counts())

print("\nTarget Mapping:")
print("0 = Malignant")
print("1 = Benign")

# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(x=df["CancerType"])
plt.title("Distribution of Cancer Types")
plt.xlabel("Cancer Type (0 = Malignant, 1 = Benign)")
plt.ylabel("Number of Records")
plt.show()

# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

# CodeNumber is an ID, so it is removed
X = df.drop(["CodeNumber", "CancerType"], axis=1)
y = df["CancerType"]

print("\nFeatures:")
print(X.columns)
print("\nNumber of Features:", X.shape[1])

# ============================================================
# 7. CORRELATION HEATMAP
# ============================================================

correlation = df.drop("CodeNumber", axis=1).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.show()

# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ============================================================
# 9. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# ============================================================
# 10. BUILD AND TRAIN MODEL
# ============================================================

model = LogisticRegression(max_iter=10000)

model.fit(X_train_scaled, y_train)

print("\nModel training completed.")

# ============================================================
# 11. PREDICTION
# ============================================================

y_pred = model.predict(X_test_scaled)

print("\nPredicted Values:")
print(y_pred)

# ============================================================
# 12. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========================================")
print("           MODEL EVALUATION")
print("========================================")
print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)

# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Malignant", "Benign"]
    )
)

# ============================================================
# 15. OBSERVATIONS AND CONCLUSION
# ============================================================

print("\n========================================")
print("       OBSERVATIONS AND CONCLUSION")
print("========================================")

print("1. The dataset contains 698 records.")
print("2. The dataset contains 9 medical features.")
print("3. CodeNumber was removed because it is only an ID.")
print("4. Missing values in BareNuclei were handled using the median.")
print("5. Features were standardized using StandardScaler.")
print("6. The dataset was divided into 80% training and 20% testing data.")
print("7. Logistic Regression was used as the classification model.")
print("8. Accuracy, Precision, Recall and F1-Score were calculated.")
print("9. The confusion matrix shows correct and incorrect predictions.")
print("10. The model predicts whether a tumor is Malignant or Benign.")
