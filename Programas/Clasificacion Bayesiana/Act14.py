# Densidad Normal y Densidad de Parzen

import numpy as np
import matplotlib.pyplot as plt
import sys


class gaussiana:

    def __init__(self, datos, sigma=0.01):
        self.datos = datos
        self.sigma = sigma
        self.d = len(datos)
        self.N = len(datos[0])

    def probabilidad_normal(self, X, M, S):
        # Encuentra la dimension del vector caracteristico
        d = len(X)
        # Evaluar la parte exponencial de la funcion normal de densidad
        e = (-1 / 2) * ((X - M) @ np.linalg.pinv(S) @ (X - M).T)
        # Obtener los eigenvalores
        V, U = np.linalg.eig(S)
        # Calculo del pseudodeterminante
        det = 1
        for i in range(0, d):
            if V[i] >= sys.float_info.epsilon:
                det = det * V[i]
        # Calculo de la constante de la funcion normal
        CTE = 1 / ((2 * np.pi) ** (d / 2) * det ** (1 / 2))
        p = CTE * np.exp(e)
        return p

    def probabilidad_parzen(self, X):
        px = 0
        for i in range(self.N):
            diff = X - self.datos[:, i]
            kernel = np.exp(-0.5 * np.dot(diff, diff) / (self.sigma**2))
            px += kernel

        px = px / (self.N * (2 * np.pi) ** (self.d / 2) * self.sigma**self.d)
        return px

    def distribucion(self, plot=False, metodo="normal"):

        # Numero de dimensiones
        d = len(np.shape(self.datos))

        # Calcular las medias
        M = np.zeros(d)
        for i in range(0, d):
            M[i] = np.mean(self.datos[i, :])

        if metodo == "normal":

            Sigma = self.sigma**2 * np.eye(self.d)

            X = np.array(self.datos[:, 0])

            x_min, x_max = np.min(self.datos[0, :]), np.max(self.datos[0, :])
            y_min, y_max = np.min(self.datos[1, :]), np.max(self.datos[1, :])

            XX, YY = np.meshgrid(
                np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
            )

            ZZ = np.zeros_like(XX)
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    point = np.array([XX[i, j], YY[i, j]])
                    ZZ[i, j] = self.probabilidad_normal(point, M, Sigma)

            title = "Grafica con funcion de densidad normal"

        elif metodo == "parzen":

            x_min, x_max = np.min(self.datos[0, :]), np.max(self.datos[0, :])
            y_min, y_max = np.min(self.datos[1, :]), np.max(self.datos[1, :])

            XX, YY = np.meshgrid(
                np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
            )

            ZZ = np.zeros_like(XX)
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    point = np.array([XX[i, j], YY[i, j]])
                    ZZ[i, j] = self.probabilidad_parzen(point)

            title = "Grafica con funcion de densidad de Parzen"

        if plot == True:

            plt.figure(1, figsize=(10, 5))
            ax = plt.axes(projection="3d")
            ax.contour3D(XX, YY, ZZ, 100)
            ax.set_xlabel("Caracteristica del eje X")
            ax.set_ylabel("Caracteristica del eje Y")
            ax.set_zlabel("Probabilidad")
            ax.set_title(title)
            plt.show()


x1 = np.array(
    [
        [-0.6705, -0.6532, -0.7735, -0.8528, -0.6322, -0.6139, -0.6731, -0.9493],
        [-0.8029, -0.7535, -0.7618, -0.7335, -0.7451, -0.6399, -0.7949, -0.8542],
    ]
)
x2 = np.array(
    [
        [-0.6628, -0.3807, -0.4556, -0.3183, -0.1202, -0.3452, -0.0667, -0.3145],
        [-0.1635, -0.3594, -0.4593, -0.3124, -0.6132, -0.6313, -0.3036, -0.5229],
    ]
)

g1 = gaussiana(x1, sigma=0.05)
g2 = gaussiana(x2, sigma=0.05)

g1.distribucion(plot=True, metodo="normal")
g1.distribucion(plot=True, metodo="parzen")
g2.distribucion(plot=True, metodo="normal")
g2.distribucion(plot=True, metodo="parzen")
