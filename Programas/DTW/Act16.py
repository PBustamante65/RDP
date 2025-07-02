# Medidas de similitud y distancia

import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

X = np.array([-2, 1, 5])
Y = np.array([-1, -2, 3])


def plot_distance(title, annotation):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Graficamos los puntos
    ax.scatter(X[0], X[1], X[2], c="g", marker="o", label="Punto X")
    ax.scatter(Y[0], Y[1], Y[2], c="b", marker="o", label="Punto Y")

    # Graficamos linea punteada desde el origen a los puntos
    x1 = np.linspace(0, X[0], 100)
    y1 = np.linspace(0, X[1], 100)
    z1 = np.linspace(0, X[2], 100)
    ax.plot(x1, y1, z1, "g--", alpha=0.5)

    x2 = np.linspace(0, Y[0], 100)
    y2 = np.linspace(0, Y[1], 100)
    z2 = np.linspace(0, Y[2], 100)
    ax.plot(x2, y2, z2, "b--", alpha=0.5)

    # Graficamos la linea entre los dos puntos
    ax.plot([X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], "r-", alpha=0.5)

    # Anotacion de distancia
    mid = (X + Y) / 2

    ax.text(mid[0], mid[1], mid[2], annotation, color="red", fontsize=12)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

    #### Distancia euclidiana


d_euc = np.sqrt(np.sum(X - Y) ** 2)
print(f"Distancia euclidiana: {d_euc}")
plot_distance("Distancia euclidiana", f"Distancia: {d_euc:.2f}")

# Distancia Manhattan (taxista)

d_man = np.sum(np.abs(X - Y))
print(f"Distancia Manhattan: {d_man}")

# Grafica distancia Manhattan
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[0], X[1], X[2], c="g", marker="o", label="Punto X")
ax.scatter(Y[0], Y[1], Y[2], c="b", marker="o", label="Punto Y")

ax.plot([X[0], Y[0]], [X[1], X[1]], [X[2], X[2]], "r-", alpha=0.5)
ax.plot([Y[0], Y[0]], [X[1], Y[1]], [X[2], X[2]], "r-", alpha=0.5)
ax.plot([Y[0], Y[0]], [Y[1], Y[1]], [X[2], Y[2]], "r-", alpha=0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Distancia Manhattan: {d_man}")
ax.legend()
plt.tight_layout()
plt.show()


# Distancia Maxima

max_val = np.zeros(len(X))
for i in range(0, len(X)):
    max_val[i] = np.abs(X[i] - Y[i])

d_max = np.max(max_val)
print(f"Distancia Maxima: {d_max}")

# Grafica de distancia maxima

max_dim = np.argmax(d_max)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[0], X[1], X[2], c="g", marker="o", label="Punto X")
ax.scatter(Y[0], Y[1], Y[2], c="b", marker="o", label="Punto Y")

if max_dim == 0:
    ax.plot([X[0], Y[0]], [X[1], X[1]], [X[2], X[2]], "r-", linewidth=4)
    ax.plot([Y[0], Y[0]], [X[1], Y[1]], [X[2], X[2]], "k-", alpha=0.3)
    ax.plot([Y[0], Y[0]], [Y[1], Y[1]], [X[2], Y[2]], "k-", alpha=0.3)
elif max_dim == 1:
    ax.plot([X[0], X[0]], [X[1], Y[1]], [X[2], X[2]], "r-", linewidth=4)
    ax.plot([X[0], Y[0]], [Y[1], Y[1]], [X[2], X[2]], "k-", alpha=0.3)
    ax.plot([Y[0], Y[0]], [Y[1], Y[1]], [X[2], Y[2]], "k-", alpha=0.3)
else:
    ax.plot([X[0], X[0]], [X[1], X[1]], [X[2], Y[2]], "r-", linewidth=4)
    ax.plot([X[0], Y[0]], [X[1], X[1]], [Y[2], Y[2]], "k-", alpha=0.3)
    ax.plot([Y[0], Y[0]], [X[1], Y[1]], [Y[2], Y[2]], "k-", alpha=0.3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Distancia Maxima (dim {['x', 'y', 'z'][max_dim]}) = {d_max}")
ax.legend()
plt.tight_layout()
plt.show()

# Distancia Minkowsky

r = 3
d_mink = (np.sum((X - Y) ** r)) ** (1 / r)
print(f"Distancia Minkowsky: {d_mink}")
plot_distance(f"Distancia Minkowski r = {r}", f"Distancia: {d_mink:.2f}")

# Distancia Mahalanobis

datos = np.array([X, Y])
mr = np.mean(X)
ms = np.mean(Y)
S = np.zeros((len(X), len(X)))
for r_idx in range(0, len(X)):
    for s in range(0, len(X)):
        suma = 0
        for i in range(0, 2):
            suma = suma + (datos[i, r_idx] - mr) * (datos[i, s] - ms)
        S[r_idx, s] = suma / 2

d_mah = np.sqrt((X - Y) @ np.linalg.pinv(S) @ (X - Y).T)
print(f"Distancia Mahalanobis: {d_mah}")
plot_distance("Distancia Mahalanobis", f"Distancia: {d_mah:.2f}")

# Distancia cuerda

Nx = np.sqrt(np.sum(X**2))
Ny = np.sqrt(np.sum(Y**2))

P = np.sum(X * Y)

d_cuerda = np.sqrt(2 - 2 * P / (Nx * Ny))
print(f"Distancia cuerda: {d_cuerda}")

# Grafica de distancia cuerda sobre una hiperesfera

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Grafica de hiperesfera

u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

# Grafica de vectores normalizados

X_norm = X / Nx
Y_norm = Y / Ny

ax.plot([0, X_norm[0]], [0, X_norm[1]], [0, X_norm[2]], "g-", linewidth=2)
ax.plot([0, Y_norm[0]], [0, Y_norm[1]], [0, Y_norm[2]], "g-", linewidth=2)

ax.scatter(X_norm[0], X_norm[1], X_norm[2], c="g", marker="o", label="X normalizado")
ax.scatter(Y_norm[0], Y_norm[1], Y_norm[2], c="b", marker="o", label="Y normalizado")

# Linea entre puntos
ax.plot(
    [X_norm[0], Y_norm[0]],
    [X_norm[1], Y_norm[1]],
    [X_norm[2], Y_norm[2]],
    "r-",
    linewidth=2,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Distancia cuerda: {d_cuerda}")
ax.legend()
plt.tight_layout()
plt.show()

# Distancia Geodesica

d_geo = np.arccos(1 - (d_cuerda / 2))
print(f"Distancia Geodesica: {d_geo}")

# Grafica de distancia cuerda sobre una hiperesfera

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Grafica de hiperesfera

u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

# Grafica de vectores normalizados

X_norm = X / Nx
Y_norm = Y / Ny

ax.plot([0, X_norm[0]], [0, X_norm[1]], [0, X_norm[2]], "g-", linewidth=2)
ax.plot([0, Y_norm[0]], [0, Y_norm[1]], [0, Y_norm[2]], "g-", linewidth=2)

ax.scatter(X_norm[0], X_norm[1], X_norm[2], c="g", marker="o", label="X normalizado")
ax.scatter(Y_norm[0], Y_norm[1], Y_norm[2], c="b", marker="o", label="Y normalizado")

# Grafico del arco geodesico
theta = np.linspace(0, np.arccos(np.sum(X_norm * Y_norm)), 50)
v = np.cross(np.cross(X_norm, Y_norm), X_norm)
v = v / np.linalg.norm(v)
arc_points = np.outer(np.cos(theta), X_norm) + np.outer(np.sin(theta), v)
ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], "r-", linewidth=2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Distancia Geodesica: {d_geo}")
ax.legend()
plt.tight_layout()
plt.show()
