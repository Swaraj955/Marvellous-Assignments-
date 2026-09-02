# ==========================================================
# CUSTOMER LOAN APPROVAL USING VOTING CLASSIFICATION
# ==========================================================

# Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score


# ==========================================================
#  Load Dataset
# ==========================================================

df = pd.read_csv("Customer_Loan_Approval.csv")

print("First 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ==========================================================
#  Check Missing Values
# ==========================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================================
#  Separate Input and Output Variables
# ==========================================================

X = df.drop("LoanApproved", axis=1)

y = df["LoanApproved"]

print("\nInput Variables:")
print(X.columns)

print("\nOutput Variable:")
print(y.name)


# ==========================================================
#  Handle Categorical Variables (if any)
# ==========================================================

# Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)


# ==========================================================
# 6. Split Dataset into Training and Testing Data
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# 7. Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================================
# 8. Create Individual Models
# ==========================================================

# Logistic Regression
logistic = LogisticRegression(
    random_state=42
)


# Decision Tree
decision_tree = DecisionTreeClassifier(
    random_state=42
)


# K-Nearest Neighbors
knn = KNeighborsClassifier(
    n_neighbors=5
)


# ==========================================================
# 9. Train Logistic Regression
# ==========================================================

logistic.fit(X_train_scaled, y_train)

logistic_prediction = logistic.predict(X_test_scaled)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

print("\nLogistic Regression Accuracy:",
      logistic_accuracy)


# ==========================================================
# 10. Train Decision Tree
# ==========================================================

decision_tree.fit(X_train, y_train)

decision_tree_prediction = decision_tree.predict(X_test)

decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_prediction
)

print("Decision Tree Accuracy:",
      decision_tree_accuracy)


# ==========================================================
# 11. Train KNN
# ==========================================================

knn.fit(X_train_scaled, y_train)

knn_prediction = knn.predict(X_test_scaled)

knn_accuracy = accuracy_score(
    y_test,
    knn_prediction
)

print("KNN Accuracy:",
      knn_accuracy)


# ==========================================================
# 12. Create Hard Voting Classifier
# ==========================================================

hard_voting = VotingClassifier(
    estimators=[
        ("logistic", logistic),
        ("decision_tree", decision_tree),
        ("knn", knn)
    ],
    voting="hard"
)


# Train Hard Voting Classifier
hard_voting.fit(
    X_train_scaled,
    y_train
)

hard_prediction = hard_voting.predict(
    X_test_scaled
)

hard_accuracy = accuracy_score(
    y_test,
    hard_prediction
)

print("\nHard Voting Accuracy:",
      hard_accuracy)


# ==========================================================
# 13. Create Soft Voting Classifier
# ==========================================================

soft_voting = VotingClassifier(
    estimators=[
        ("logistic", logistic),
        ("decision_tree", decision_tree),
        ("knn", knn)
    ],
    voting="soft"
)


# Train Soft Voting Classifier
soft_voting.fit(
    X_train_scaled,
    y_train
)

soft_prediction = soft_voting.predict(
    X_test_scaled
)

soft_accuracy = accuracy_score(
    y_test,
    soft_prediction
)

print("Soft Voting Accuracy:",
      soft_accuracy)


# ==========================================================
# 14. Final Comparison Table
# ==========================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Hard Voting",
        "Soft Voting"
    ],

    "Accuracy": [
        logistic_accuracy,
        decision_tree_accuracy,
        knn_accuracy,
        hard_accuracy,
        soft_accuracy
    ]
})


print("\n======================================")
print("FINAL MODEL COMPARISON")
print("======================================")

print(comparison)


# ==========================================================
# 15. Accuracy in Percentage
# ==========================================================

comparison["Accuracy (%)"] = (
    comparison["Accuracy"] * 100
).round(2)


print("\nAccuracy in Percentage:")
print(comparison[
    ["Model", "Accuracy (%)"]
])


# ==========================================================
# 16. Find Best Model
# ==========================================================

best_model = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print("\n======================================")
print("BEST MODEL")
print("======================================")

print("Model:",
      best_model["Model"])

print("Accuracy:",
      best_model["Accuracy (%)"],
      "%")