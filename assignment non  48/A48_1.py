X = [1, 2, 3, 4, 5]
Y = [3, 4, 2, 4, 5]

# Mean
mean_x = sum(X) / len(X)
mean_y = sum(Y) / len(Y)

# Calculate slope
numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(X, Y))
denominator = sum((x - mean_x) ** 2 for x in X)

m = numerator / denominator

# Calculate intercept
c = mean_y - m * mean_x

# Prediction
x_new = 6
y_pred = m * x_new + c

print("Mean of X =", mean_x)
print("Mean of Y =", mean_y)
print("Slope (m) =", m)
print("Intercept (c) =", c)
print("Regression Equation: Y =", m, "X +", c)
print("Predicted Y for X = 6:", y_pred)