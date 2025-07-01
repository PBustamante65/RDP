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

    ax.scatter(X[0], X[1], X[2], c="g", marker="o", label="Punto X")
    ax.scatter(Y[0], Y[1], Y[2], c="b", marker="o", label="Punto Y")

    x1 = np.linspace(0, X[0], 100)
    y1 = np.linspace(0, X[1], 100)
    z1 = np.linspace(0, X[2], 100)
    ax.plot(x1, y1, z1, "g--", alpha=0.5)

    x2 = np.linspace(0, Y[0], 100)
    y2 = np.linspace(0, Y[1], 100)
    z2 = np.linspace(0, Y[2], 100)
