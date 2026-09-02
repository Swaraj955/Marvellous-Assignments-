
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. Load Dataset
# ============================================================

# Change this filename according to your dataset
df = pd.read_csv("Fraudulent_Transaction_Detection.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. Data Preprocessing
# ============================================================

# Remove unnecessary ID column if present
columns_to_drop = []

if "Transaction_ID" in df.columns:
    columns_to_drop.append("Transaction_ID")

if "TransactionID" in df.columns:
    columns_to_drop.append("TransactionID")

if len(columns_to_drop) > 0:
    df = df.drop(columns=columns_to_drop)


# ------------------------------------------------------------
# Encode categorical columns
# ------------------------------------------------------------

# Device Type is generally categorical
if "Device Type" in df.columns:
    df["Device Type"] = df["Device Type"].astype("category").cat.codes

# Alternative column name
if "Device_Type" in df.columns:
    df["Device_Type"] = df["Device_Type"].astype("category").cat.codes


# Encode any remaining object/string columns
for column in df.select_dtypes(include=["object"]).columns:

    # Don't encode target here
    if column != "Fraud":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))


# ============================================================
# 4. Separate Features and Target
# ============================================================

X = df.drop("Fraud", axis=1)
y = df["Fraud"]


print("\nFeatures:")
print(X.columns)

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# 5. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 6. Create Machine Learning Models
# ============================================================

# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=8
)


# ------------------------------------------------------------
# Bagging Classifier
# ------------------------------------------------------------

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),
    n_estimators=100,
    random_state=42
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


# ------------------------------------------------------------
# AdaBoost
# ------------------------------------------------------------

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=2,
        random_state=42
    ),
    n_estimators=100,
    learning_rate=0.8,
    random_state=42
)


# ------------------------------------------------------------
# Voting Classifier
# ------------------------------------------------------------

voting = VotingClassifier(
    estimators=[
        ("decision_tree", decision_tree),
        ("random_forest", random_forest),
        ("adaboost", adaboost)
    ],
    voting="hard"
)


# ============================================================
# 7. Store Models
# ============================================================

models = {
    "Decision Tree": decision_tree,
    "Bagging": bagging,
    "Random Forest": random_forest,
    "AdaBoost": adaboost,
    "Voting": voting
}


# ============================================================
# 8. Train and Evaluate Models
# ============================================================

results = []

confusion_matrices = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    confusion_matrices[name] = cm

    # Store results
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Print results
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ============================================================
# 9. Final Comparison Table
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(results_df.to_string(index=False))


# ============================================================
# 10. Convert Scores to Percentage
# ============================================================

comparison_percentage = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:
    comparison_percentage[column] = (
        comparison_percentage[column] * 100
    ).round(2)

print("\n\nComparison in Percentage:")
print(comparison_percentage.to_string(index=False))


# ============================================================
# 11. Find Best Model
# ============================================================

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Algorithm :", best_model["Algorithm"])
print("Accuracy  :", round(best_model["Accuracy"], 4))
print("Precision :", round(best_model["Precision"], 4))
print("Recall    :", round(best_model["Recall"], 4))
print("F1 Score  :", round(best_model["F1 Score"], 4))


# ============================================================
# 12. Plot Model Comparison
# ============================================================

plot_df = results_df.set_index("Algorithm")

plot_df[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Algorithm")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ============================================================
# 13. Display Confusion Matrices
# ============================================================

for name, cm in confusion_matrices.items():

    plt.figure(figsize=(5, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )

    disp.plot()
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ============================================================
# 14. Save Final Comparison
# ============================================================

comparison_percentage.to_csv(
    "fraud_model_comparison.csv",
    index=False
)

print("\nComparison saved as:")
print("fraud_model_comparison.csv")