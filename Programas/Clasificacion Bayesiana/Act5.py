# Funcion discriminante multivariable

import numpy as np
import matplotlib.pyplot as plt


# Funcion discriminante multivariable
def disc(x, mu, cov, priori, dim=2):
    a = -1 / 2 * (x - mu).T @ np.linalg.inv(cov) @ (x - mu)
    b = -(dim / 2) * np.log(2 * np.pi)
    c = -1 / 2 * np.log(np.linalg.det(cov))
    d = np.log(priori)
    return a + b + c + d


# Clase 1
A1 = np.array([3, 0])
A2 = np.array([5, -2])
A3 = np.array([3, -4])
A4 = np.array([1, -2])

# Clase 2
B1 = np.array([3, 8])
B2 = np.array([4, 6])
B3 = np.array([3, 4])
B4 = np.array([2, 6])

A = np.array([A1, A2, A3, A4])
B = np.array([B1, B2, B3, B4])

plt.figure(1, figsize=(10, 5))
plt.plot(A[:, 0], A[:, 1], "*", label="Clase 1")
plt.plot(B[:, 0], B[:, 1], "x", label="Clase 2")
plt.title("Diagrama de dispersion")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

mean_A = np.mean(A, axis=0)
mean_B = np.mean(B, axis=0)

x = np.array([float(input("Ingrese x1: ")), float(input("Ingrese x2: "))])

plt.figure(2, figsize=(10, 5))
plt.plot(A[:, 0], A[:, 1], "*", label="Clase 1")
plt.plot(B[:, 0], B[:, 1], "x", label="Clase 2")
plt.scatter(mean_A[0], mean_A[1], label="Media de A")
plt.scatter(mean_B[0], mean_B[1], label="Media de B")
plt.scatter(x[0], x[1], marker="D", color="black", label="Punto de prueba")
plt.title("Diagrama de dispersion")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

cov_A = np.cov(A.T)
cov_B = np.cov(B.T)

disc_A = disc(x, mean_A, cov_A, 0.20)
disc_B = disc(x, mean_B, cov_B, 0.80)

print(f"Discriminante Clase 1: {disc_A}")
print(f"Discriminante Clase 2: {disc_B}")

if disc_A > disc_B:
    print("El punto pertenece a la clase 1")
else:
    print("El punto pertenece a la clase 2")


# Region de decision
# Utilizando la funcion discriminante y los datos de cada clase (media, covarianza y priori) se calcula la region de decision
xr = np.linspace(0, 5, 100)  # Se generan 100 puntos entre 0 y 5
yr = 0.0033 * (
    1033 - 338 * xr + 57 * xr**2
)  # Utilizando matematica, se despejan las ecuaciones para obtener la funcion de la region de decision

plt.figure(3, figsize=(10, 5))
plt.plot(
    A[:, 0], A[:, 1], "*", label="Clase 1"
)  # Graficamos todos los puntos de x (A[:,0], todos los puntos de y (A[:,1])
plt.plot(B[:, 0], B[:, 1], "x", label="Clase 2")
plt.scatter(
    mean_A[0], mean_A[1], label="Media de A"
)  # Calculamos la media de cada clase
plt.scatter(mean_B[0], mean_B[1], label="Media de B")
plt.scatter(
    x[0], x[1], marker="D", color="black", label="Punto de prueba"
)  # Mostramos el punto de prueba
plt.plot(xr, yr, label="Region de decision")  # Mostramos la region de decision
plt.title("Diagrama de dispersion")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
