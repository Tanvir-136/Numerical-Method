# 1. Linear Regression

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

def gradient_descent(x, y, m = 0, b = 0, learning_rate = 0.01, epochs = 10000):
    n = len(y)
    for _ in range(epochs):
        y_pred = m * x + b
        dm = (-2/n) * sum(x * (y - y_pred))
        db = (-2/n) * sum(y - y_pred)
        m -= learning_rate * dm
        b -= learning_rate * db
    return m, b

plt.scatter(x, y)

m, b = gradient_descent(x, y)
plt.plot(x, m*x + b, color='red')

plt.xlabel('X')
plt.ylabel('Y')

plt.title('Gradient Descent')
plt.show()

# 2. Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 1 + 2*x + 3*x**2 + np.random.randn(100, 1)

def poly_gradient_descent(x, y, a0=0, a1=0, a2=0, learning_rate=0.01, epochs=10000):
    n = len(y)
    for _ in range(epochs):
        y_pred = a0 + a1*x + a2*x**2
        
        # Gradients
        da0 = (-2/n) * sum(y - y_pred)
        da1 = (-2/n) * sum((y - y_pred) * x)
        da2 = (-2/n) * sum((y - y_pred) * x**2)
        
        # Update parameters
        a0 -= learning_rate * da0
        a1 -= learning_rate * da1
        a2 -= learning_rate * da2
        
    return a0, a1, a2

a0, a1, a2 = poly_gradient_descent(x, y)

# Predicted curve
x_plot = np.linspace(0, 2, 100).reshape(-1,1)
y_plot = a0 + a1*x_plot + a2*x_plot**2

plt.scatter(x, y, label='Data points')
plt.plot(x_plot, y_plot, color='red', label='Polynomial fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression via Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

print(f"a0 = {a0}, a1 = {a1}, a2 = {a2}")


# Logistic Regression
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Generate Sample Data ----------
np.random.seed(0)
x = np.random.rand(100, 1) * 10       # feature
y = (x > 5).astype(int)               # label: 0 if x<=5, 1 if x>5

# ---------- 2. Sigmoid Function ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------- 3. Gradient Descent ----------
m, b = 0, 0           
lr = 0.01             
epochs = 1000
n = len(y)

for _ in range(epochs):
    z = m*x + b
    y_pred = sigmoid(z)
    dm = (1/n) * sum((y_pred - y) * x)
    db = (1/n) * sum(y_pred - y)
    m -= lr * dm
    b -= lr * db

# ---------- 4. Plot Data and Sigmoid ----------
x_plot = np.linspace(0, 10, 100).reshape(-1,1)
y_plot = sigmoid(m*x_plot + b)

plt.scatter(x, y, c=y, cmap='bwr')
plt.plot(x_plot, y_plot, color='black', label='Sigmoid curve')
plt.xlabel("X")
plt.ylabel("Probability")
plt.title("Logistic Regression via Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# ---------- 5. Predict Function ----------
def predict(x_input):
    return (sigmoid(m*x_input + b) >= 0.5).astype(int)

print("Example Predictions for [2, 6, 8]:", predict(np.array([[2],[6],[8]])))
