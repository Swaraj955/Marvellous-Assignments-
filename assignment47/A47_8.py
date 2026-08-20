from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [50, 55, 60, 65, 70]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])