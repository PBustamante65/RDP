# Ventanas de Parzen

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Muestras 1
x1 = np.array(
    [
        -0.6705,
        -0.6532,
        -0.7735,
        -0.8528,
        -0.6322,
        -0.6139,
        -0.6731,
        -0.9493,
        -0.8029,
        -0.7535,
        -0.7618,
        -0.7335,
        -0.7451,
        -0.6399,
        -0.7949,
        -0.8542,
        -0.7634,
        -0.6402,
        -0.7409,
    ]
)

# Muestras 2

x2 = np.array(
    [
        -0.6628,
        -0.3807,
        -0.4556,
        -0.3183,
        -0.1202,
        -0.3452,
        -0.0667,
        -0.3145,
        -0.1635,
        -0.3594,
        -0.4593,
        -0.3124,
        -0.6132,
        -0.6313,
        -0.3036,
        -0.5229,
        -0.4250,
        -0.3504,
        -0.406,
    ]
)

# Definimos un arreglo de ceros para graficar las muestras

y = np.zeros(len(x1))

# Graficar las muestras

plt.figure(1, figsize=(10, 5))
plt.plot(x1, y, "o")
plt.plot(x2, y, "*")
plt.title("Distribucion de muestras")
plt.xlabel("Muestras")
plt.show()

# Valor minimo y maximo para evaluar la densidad resultante

vmin = -1.5
vmax = 0.5

# Contamos el numero de muestras de cada clase
N1 = len(x1)
N2 = len(x2)

# Tamaño de la ventana (kernel) dado por la varianza para ambas clases
s1 = 0.01
s2 = s1

# Resoulucion a utilizar para generar valores del eje x
inc = 0.1
# Generamos valores en el eje x desde vmin al vmax
v_x = np.arange(vmin, vmax, inc)

# Evaluacion del kernel gaussiano para la clase 1
py = np.zeros((N1, len(v_x)))
k = 0
for i in range(0, N1):
    for j in range(0, len(v_x)):
        py[i, j] = (1 / (s1 * np.sqrt(2 * np.pi))) * np.exp(
            -((v_x[j] - x1[i]) ** 2) / (2 * s1**2)
        )

plt.figure(2, figsize=(10, 5))
# Graficamos la gaussiana de la clase 1
for i in range(0, N1):
    plt.plot(v_x, py[i, :], "r")


# Evaluacion del kernel gaussiano para la clase 2

py2 = np.zeros((N2, len(v_x)))
k = 0
for i in range(0, N2):
    for j in range(0, len(v_x)):
        py2[i, j] = (1 / (s2 * np.sqrt(2 * np.pi))) * np.exp(
            -((v_x[j] - x2[i]) ** 2) / (2 * s2**2)
        )

# Graficamos la gaussiana de la clase 2
for i in range(0, N2):
    plt.plot(v_x, py2[i, :], "b")
plt.title("Grafica de las densidades")
plt.show()


# Densidad de probabilidad resultante de la clase 1 y 2
px1 = np.zeros(len(v_x))
px2 = np.zeros(len(v_x))

for i in range(0, N1):
    px1 = px1 + py[i, :]
for i in range(0, N2):
    px2 = px2 + py2[i, :]

# Normalizacion de la densidad

px1 = px1 / N1 * inc
px2 = px2 / N2 * inc

# Grafica de la densidad resultante

plt.figure(3, figsize=(10, 5))
plt.plot(v_x, px1, "r")
plt.plot(v_x, px2, "b")
plt.title("Estimacion de la densidad por ventanas de Parzen")
plt.xlabel("X")
plt.ylabel("P(xi)")
plt.show()
