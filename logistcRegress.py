import numpy as np
# Linear regression is one of the most fundamental techniques in statistics and machine learning. It aims to model the relationship between a dependent variable
# y and one or more independent variables X.

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X, y):
        # initialize the weights and bias
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randint(1)
        # Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            gradient = self.gradient(X, y, y_predicted)
            self.weights -= self.lr * gradient[0]
            self.bias -= self.lr * gradient[1]
        return self

    def gradient(self, X, y, y_predicted):
        n_samples, n_features = X.shape
        # Calculate the gradient of the cost function with respect to the weights and bias
        gradient_weights = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        gradient_bias = (1 / n_samples) * np.sum(y_predicted - y)
        return gradient_weights, gradient_bias

    def predict(self, X):
        # check if the user called the fit method
        if self.weights is None:
            raise ValueError("You must call the fit method before calling the predict method")
        # calculate the predictions
        linear_model = np.dot(X, self.weights) + self.bias
        return linear_model

X = np.linspace(-40, 40, 10000).reshape(-1, 1)  # 100 points between -2 and 2
y = X.T.reshape(-1) + 5


model = LinearRegression()

model.fit(X,y)

print(model.bias)
