# Linear Regression from Scratch in Python

## Overview
This project implements **Linear Regression** from scratch using NumPy. It supports **L1 (Lasso) and L2 (Ridge) regularization** and uses **Gradient Descent** for optimization. The model also includes early stopping based on loss convergence.

## Features
- Implements **Batch Gradient Descent** for learning weights.
- Supports **L1 (Lasso) and L2 (Ridge) regularization**.
- Uses **Mean Squared Error (MSE)** as the loss function.
- Includes a **learning rate** and **early stopping** based on tolerance.
- Provides a `fit` method for training and a `predict` method for inference.

## Implementation Details
- **Loss Function**:
  $$\text{MSE Loss} = \frac{1}{n} \sum (y - \hat{y})^2$$
  - With **L1 Regularization**: $$ \text{Loss} += \alpha \sum |w| $$
  - With **L2 Regularization**: $$ \text{Loss} += \alpha \sum w^2 $$

- **Gradient Calculation**:
  - For **weights**:
    $$\nabla w = \frac{1}{n} X^T (\hat{y} - y) + \alpha \cdot \text{Regularization Term}$$
  - For **bias**:
    $$\nabla b = \frac{1}{n} \sum (\hat{y} - y)$$


## How to Use
```python
# Generate dummy data
import numpy as np
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1

# Train model
model = LinearRegression(learning_rate=0.01, n_iters=1000, penalty='l2')
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
```

## Future Enhancements
- Add **Stochastic Gradient Descent (SGD)** for faster convergence.
- Implement **Polynomial Regression** by adding feature transformation.

## Conclusion
This project builds **Linear Regression from scratch**, explaining the impact of **L1 and L2 regularization** on coefficients. It's a solid foundation for understanding **machine learning optimization** techniques.

---
🛠️ *Built with NumPy and Python* 🚀

