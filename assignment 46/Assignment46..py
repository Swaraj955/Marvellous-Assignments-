
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Get Data


data = pd.read_csv("Advertising (1).csv")

# Display dataset
print("Dataset:")
print(data)

# Step 2: Clean, Prepare and Manipulate Data

X = data[['TV', 'radio', 'newspaper']]
Y = data['sales']

# Step 3: Train Data

# 50% data for training and 50% for testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=42
)

# Create Linear Regression model

model = LinearRegression()

# Train the model

model.fit(X_train, Y_train)

# Step 4: Test the Data

Y_pred = model.predict(X_test)

# Step 5: Display Predicted and Expected Values

print("\nPredicted Sales and Expected Sales:")

for predicted, expected in zip(Y_pred, Y_test):
    print("Predicted:", round(predicted, 2),
          "Expected:", expected)