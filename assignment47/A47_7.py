from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [50, 55, 60, 65, 70]

model = LinearRegression()
model.fit(X, y)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)