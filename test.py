from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from linearRegress import LinearRegression, MSE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
# X, y = fetch_california_housing(return_X_y=True)
X = np.linspace(0, 40, 100000).reshape(-1, 1)  # 100,000 points, one feature
X = np.hstack([X, X**2])  # Adding a second feature (e.g., X²)

# print(X.shape)

# Generate y
y = X[:, 0] + 5  # Only using the first feature + bias term
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()  # Create an instance of LinearRegression
model.fit(X_train, y_train)  # Train the model on training data
y_pred = model.predict(X_test)  # Predict on test data


mse = mean_squared_error(y_test, y_pred)  # Compute Mean Squared Error
print(f"Mean Squared Error: {mse}")

# Print the learned parameters
print(f"Intercept (bias): {model.intercept_}")
print(f"Coefficient (weights): {model.coef_}")
print(y_pred)

# model = LinearRegression(n_iters=10000)

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print(MSE(y_pred, y_test))
