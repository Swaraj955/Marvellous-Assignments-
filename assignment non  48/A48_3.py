import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = [[1], [2], [3], [4], [5]]
Y = [20000, 25000, 30000, 35000, 40000]

# Create model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Predict salary for 6 years
prediction = model.predict([[6]])

print("Predicted Salary for 6 Years Experience: ", prediction[0])

# Predict values for graph
Y_pred = model.predict(X)

# Plot data points
plt.scatter(X, Y, label="Data Points")

# Plot regression line
plt.plot(X, Y_pred, label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
plt.show()