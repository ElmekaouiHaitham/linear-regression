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

model = LinearRegression(n_iters=10000, learning_rate=0.001)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(MSE(y_pred, y_test))
