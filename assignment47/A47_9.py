from sklearn.linear_model import LinearRegression

X = [
    [1, 7],
    [2, 6],
    [3, 7],
    [4, 6],
    [5, 8]
]

y = [50, 55, 60, 65, 70]

model = LinearRegression()
model.fit(X, y)

print("Coefficient for Study Hours:", model.coef_[0])
print("Coefficient for Sleep Hours:", model.coef_[1])
print("Intercept:", model.intercept_)