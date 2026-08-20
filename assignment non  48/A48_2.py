X = [1, 2, 3, 4, 5]
Y = [3, 4, 2, 4, 5]

m = 0.4
c = 2.4

predicted = []

for x in X:
    y_pred = m * x + c
    predicted.append(y_pred)

print("Actual Y     Predicted Y     Squared Error")

sse = 0

for actual, pred in zip(Y, predicted):
    error = actual - pred
    squared_error = error ** 2
    sse += squared_error
    print(actual, "          ", pred, "          ", squared_error)

mse = sse / len(Y)

mean_y = sum(Y) / len(Y)

sst = sum((y - mean_y) ** 2 for y in Y)

r2 = 1 - (sse / sst)

print("\nMSE =", mse)
print("R2 Score =", r2)