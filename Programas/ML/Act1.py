# Regresion Lineal

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, label="Datos", s=20)
plt.xlabel("X")
plt.ylabel("Y")

X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, "r-")
plt.grid(True)
plt.show()


#####Libreria Sklearn

lin_reg = LinearRegression()
lin_reg.fit(X, y)
predict = lin_reg.predict(X_new)
print("\n")
print(f"Sklearn: {predict}")
