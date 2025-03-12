from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linearRegress import LinearRegression, MSE
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

import numpy as np
X, y = fetch_california_housing(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sgd = SGDRegressor(learning_rate="constant", eta0=0.001, max_iter=1000, tol=1e-3, random_state=42)
# sgd.fit(X_train, y_train)

# y_pred = sgd.predict(X_test)
# from sklearn.metrics import mean_squared_error, r2_score

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"MSE: {mse:.2f}")
# print(f"R²: {r2:.2f}")


model = LinearRegression(n_iters=10000, learning_rate=0.001)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(MSE(y_pred, y_test))
