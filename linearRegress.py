import numpy as np
# Linear regression is one of the most fundamental techniques in statistics and machine learning. It aims to model the relationship between a dependent variable
# y and one or more independent variables X.

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, tol=0.001, penalty = None):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None
        self.penalty = penalty

    def loss(self, y, y_pred):
        loss = np.mean((y - y_pred) ** 2)
        if self.penalty == 'l1':
            loss += self.lr * np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            loss += self.lr * np.sum(self.weights ** 2)
        return loss

    def fit(self, X, y):
        # initialize the weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.random.randn()
        prv_loss = None
        # Gradient Descent
        for _ in range(self.n_iters):
            print(f"step{_}:")
            y_predicted = self.predict(X)
            gradient = self.__gradient(X, y, y_predicted)
            self.weights -= self.lr * gradient[0]
            self.bias -= self.lr * gradient[1]
            loss = self.loss(y, self.predict(X))
            print("loss= ", loss)
            if prv_loss is not None and abs(loss - prv_loss) < self.tol:
                break
            prv_loss = loss
        return self

    def __gradient(self, X, y, y_predicted):
        n_samples, _ = X.shape
        # Calculate the gradient of the cost function with respect to the weights and bias
        gradient_weights = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        if self.penalty == 'l1':
            gradient_weights += self.lr * np.sign(self.weights)
        elif self.penalty == 'l2':
            gradient_weights += self.lr * 2 * self.weights
        gradient_bias = (1 / n_samples) * np.sum(y_predicted - y)
        return gradient_weights, gradient_bias

    def predict(self, X):
        # check if the user called the fit method
        if self.weights is None:
            raise ValueError("You must call the fit method before calling the predict method")
        # calculate the predictions
        pred = np.dot(X, self.weights) + self.bias
        return pred

def MSE(y_pred, y_true):
    return np.mean((y_true - y_pred) ** 2)

