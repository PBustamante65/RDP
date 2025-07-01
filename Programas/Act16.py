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
