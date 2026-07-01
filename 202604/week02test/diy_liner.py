import numpy as np

x = np.array([1.0, 2.0, 3.0])

w = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

b = np.array([0.1, -0.1])

y = w @ x + b
print(y)
print(w * x + b)
